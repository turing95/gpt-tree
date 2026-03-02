#!/usr/bin/env python3
"""RR-TreeLM v4 reference implementation.

Implements the public CLI:
  - prepare
  - ppm-gate
  - build-features
  - train-cascade
  - build-retrieval
  - train-calibrator
  - eval
  - sample
  - benchmark
  - benchmark-tf-shard
  - benchmark-assemble

The code is intentionally explicit and reproducible rather than minimal.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import random
import re
import shutil
import string
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from sklearn.utils import murmurhash3_32
except Exception:  # pragma: no cover
    murmurhash3_32 = None

try:  # pragma: no cover
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
CONFIG_DIR = ROOT / "configs"

DATASET_CHOICES = ("tinyshakespeare", "wikitext2_char", "enwik8_10mb")

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
WIKITEXT2_FILES = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

DEFAULT_SEEDS = (1337, 2027, 9001)
DEFAULT_PROMPTS = ("", "ROMEO:", "JULIET:")


@dataclass
class FeatureSpec:
    ctx_len: int = 256
    onehot_vocab_bins: int = 97
    pos_onehot_dim: int = 24832
    pos_ngram_dim: int = 49152
    skipgram_dim: int = 24576
    misc_dim: int = 2048
    hash_collision_threshold: float = 0.2
    adjusted_namespaces: Dict[str, bool] = field(default_factory=dict)
    total_dim: int = 100608

    def recompute_total(self) -> None:
        self.total_dim = self.pos_onehot_dim + self.pos_ngram_dim + self.skipgram_dim + self.misc_dim


@dataclass
class StageParams:
    depth: int
    iterations: int
    learning_rate: float
    l2_leaf_reg: float
    rsm: float


@dataclass
class CascadeConfig:
    k_folds: int = 3
    thread_count: int = 8
    random_seed: int = 1337
    stage1: StageParams = field(
        default_factory=lambda: StageParams(depth=7, iterations=1200, learning_rate=0.05, l2_leaf_reg=8.0, rsm=0.25)
    )
    stage2: StageParams = field(
        default_factory=lambda: StageParams(depth=6, iterations=900, learning_rate=0.05, l2_leaf_reg=10.0, rsm=0.25)
    )
    stage3: StageParams = field(
        default_factory=lambda: StageParams(depth=6, iterations=700, learning_rate=0.04, l2_leaf_reg=12.0, rsm=0.25)
    )


@dataclass
class FusionWeights:
    ppm_alpha: float = 1.0
    stage_beta: float = 0.35
    simhash_gamma: float = 0.08
    overlap_delta: float = 0.08
    calibrator_epsilon: float = 0.2
    fusion_mode: str = "ppm_primary"
    calibrator_mode: str = "delta"
    calibrator_clamp: float = 0.35
    aux_depth_threshold: float = 6.0
    aux_entropy_threshold: float = 3.0


def normalize_fusion_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    # Backward compatibility with v3 naming.
    if "a" in payload or "b1" in payload or "b2" in payload or "b3" in payload or "c" in payload:
        mapped = {
            "ppm_alpha": float(payload.get("b1", 0.5)),
            "stage_beta": float(payload.get("a", 1.0)),
            "simhash_gamma": float(payload.get("b2", 0.2)),
            "overlap_delta": float(payload.get("b3", 0.2)),
            "calibrator_epsilon": float(payload.get("c", 0.4)),
            "fusion_mode": payload.get("fusion_mode", "balanced"),
            "calibrator_mode": payload.get("calibrator_mode", "full"),
            "calibrator_clamp": float(payload.get("calibrator_clamp", 0.35)),
            "aux_depth_threshold": float(payload.get("aux_depth_threshold", 6.0)),
            "aux_entropy_threshold": float(payload.get("aux_entropy_threshold", 3.0)),
        }
        return mapped
    return dict(payload)


def fusion_from_args(args: argparse.Namespace, fallback: Optional[FusionWeights] = None) -> FusionWeights:
    base = fallback or FusionWeights()
    def pick(name: str, default: Any) -> Any:
        val = getattr(args, name, None)
        return default if val is None else val
    return FusionWeights(
        ppm_alpha=float(pick("ppm_alpha", base.ppm_alpha)),
        stage_beta=float(pick("stage_beta", base.stage_beta)),
        simhash_gamma=float(pick("simhash_gamma", base.simhash_gamma)),
        overlap_delta=float(pick("overlap_delta", base.overlap_delta)),
        calibrator_epsilon=float(pick("calibrator_epsilon", base.calibrator_epsilon)),
        fusion_mode=str(pick("fusion_mode", base.fusion_mode)),
        calibrator_mode=str(pick("calibrator_mode", base.calibrator_mode)),
        calibrator_clamp=float(pick("calibrator_clamp", base.calibrator_clamp)),
        aux_depth_threshold=float(pick("aux_depth_threshold", base.aux_depth_threshold)),
        aux_entropy_threshold=float(pick("aux_entropy_threshold", base.aux_entropy_threshold)),
    )


@dataclass
class RuntimeMeta:
    thread_count: int
    seed: int
    created_at: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def _file_lock(path: Path):
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def dataset_dir(dataset: str) -> Path:
    return DATA_DIR / dataset


def artifact_dir(dataset: str) -> Path:
    return ART_DIR / dataset


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def physical_cores() -> int:
    cores = os.cpu_count() or 4
    return max(1, cores // 2) if cores > 2 else cores


def default_thread_count() -> int:
    return min(8, physical_cores())


def set_global_thread_env(thread_count: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)


def parse_csv_ints(value: str) -> List[int]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def build_env_metadata(thread_count: Optional[int] = None) -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical_estimate": physical_cores(),
        "thread_count_frozen": int(thread_count if thread_count is not None else default_thread_count()),
        "cwd": str(Path.cwd()),
        "created_at": now_iso(),
    }


def record_run_step(dataset: str, command: str, args: argparse.Namespace, extra: Optional[Mapping[str, Any]] = None) -> None:
    adir = artifact_dir(dataset)
    ensure_dir(adir)
    mpath = adir / "run_manifest.json"
    lock_path = adir / "run_manifest.lock"
    with _file_lock(lock_path):
        if mpath.exists():
            try:
                payload = load_json(mpath)
            except Exception:
                payload = {
                    "dataset": dataset,
                    "created_at": now_iso(),
                    "environment": build_env_metadata(),
                    "commands": [],
                }
        else:
            payload = {
                "dataset": dataset,
                "created_at": now_iso(),
                "environment": build_env_metadata(),
                "commands": [],
            }

        def _safe(v: Any) -> Any:
            if callable(v):
                return None
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {str(k): _safe(val) for k, val in v.items() if not callable(val)}
            if isinstance(v, (list, tuple)):
                return [_safe(x) for x in v if not callable(x)]
            return str(v)

        safe_args = {k: _safe(v) for k, v in vars(args).items() if k != "fn" and not callable(v)}
        entry = {
            "command": command,
            "args": safe_args,
            "timestamp": now_iso(),
        }
        if extra:
            entry["extra"] = dict(extra)
        payload.setdefault("commands", []).append(entry)
        save_json(mpath, payload)


def write_frozen_config(dataset: str, payload: Mapping[str, Any]) -> None:
    adir = artifact_dir(dataset)
    ensure_dir(adir)
    save_json(adir / "frozen_config.json", dict(payload))


def update_manifest(dataset: str, updates: Mapping[str, Any]) -> None:
    adir = artifact_dir(dataset)
    ensure_dir(adir)
    mpath = adir / "run_manifest.json"
    lock_path = adir / "run_manifest.lock"
    with _file_lock(lock_path):
        if mpath.exists():
            try:
                payload = load_json(mpath)
            except Exception:
                payload = {"dataset": dataset, "created_at": now_iso(), "environment": build_env_metadata(), "commands": []}
        else:
            payload = {"dataset": dataset, "created_at": now_iso(), "environment": build_env_metadata(), "commands": []}
        payload.update(dict(updates))
        save_json(mpath, payload)


ASCII_CHARS = "\n" + "".join(chr(i) for i in range(32, 127))


class Vocabulary:
    """Fixed vocabulary: BOS + printable ascii/newline + UNK."""

    def __init__(self) -> None:
        self.bos_token = "<BOS>"
        self.unk_token = "<UNK>"
        self.id_to_token: List[str] = [self.bos_token]
        self.id_to_token.extend(list(ASCII_CHARS))
        self.id_to_token.append(self.unk_token)
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.id_to_token)}
        self.bos_id = self.token_to_id[self.bos_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.base_ascii_bins = 97  # BOS + 96 ascii/newline bins

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> np.ndarray:
        out = np.empty(len(text), dtype=np.int16)
        for i, ch in enumerate(text):
            out[i] = self.token_to_id.get(ch, self.unk_id)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        chars: List[str] = []
        for idx in ids:
            if idx == self.bos_id:
                continue
            tok = self.id_to_token[int(idx)]
            if tok in (self.bos_token, self.unk_token):
                chars.append("?")
            else:
                chars.append(tok)
        return "".join(chars)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id_to_token": self.id_to_token,
            "bos_id": self.bos_id,
            "unk_id": self.unk_id,
            "base_ascii_bins": self.base_ascii_bins,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Vocabulary":
        obj = cls()
        obj.id_to_token = list(payload["id_to_token"])
        obj.token_to_id = {tok: i for i, tok in enumerate(obj.id_to_token)}
        obj.bos_id = int(payload["bos_id"])
        obj.unk_id = int(payload["unk_id"])
        obj.base_ascii_bins = int(payload.get("base_ascii_bins", 97))
        return obj


def download_text(url: str) -> str:
    import ssl
    from urllib.request import urlopen

    try:
        with urlopen(url) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        # Some local environments have incomplete cert chains; fallback keeps CLI usable.
        ctx = ssl._create_unverified_context()
        with urlopen(url, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="replace")


def prepare_dataset(dataset: str) -> None:
    ddir = dataset_dir(dataset)
    ensure_dir(ddir)
    raw_path = ddir / "raw.txt"

    if dataset == "tinyshakespeare":
        if not raw_path.exists():
            text = download_text(TINY_SHAKESPEARE_URL)
            raw_path.write_text(text, encoding="utf-8")
    elif dataset == "wikitext2_char":
        if not raw_path.exists():
            parts = []
            for split_name, url in WIKITEXT2_FILES.items():
                txt = download_text(url)
                parts.append(f"\n\n### {split_name.upper()} ###\n\n")
                parts.append(txt)
            raw_path.write_text("".join(parts), encoding="utf-8")
    elif dataset == "enwik8_10mb":
        if not raw_path.exists():
            from urllib.request import urlopen
            from zipfile import ZipFile
            import io
            import ssl

            try:
                with urlopen(ENWIK8_URL) as resp:
                    zbytes = resp.read()
            except Exception:
                ctx = ssl._create_unverified_context()
                with urlopen(ENWIK8_URL, context=ctx) as resp:
                    zbytes = resp.read()
            with ZipFile(io.BytesIO(zbytes)) as zf:
                with zf.open("enwik8", "r") as ef:
                    content = ef.read(10_000_000)
            raw_path.write_text(content.decode("utf-8", errors="replace"), encoding="utf-8")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    text = raw_path.read_text(encoding="utf-8", errors="replace")
    n = len(text)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    (ddir / "train.txt").write_text(text[:train_end], encoding="utf-8")
    (ddir / "val.txt").write_text(text[train_end:val_end], encoding="utf-8")
    (ddir / "test.txt").write_text(text[val_end:], encoding="utf-8")

    vocab = Vocabulary()
    adir = artifact_dir(dataset)
    ensure_dir(adir)
    save_json(adir / "vocab.json", vocab.to_json())
    save_json(
        adir / "splits.json",
        {
            "dataset": dataset,
            "raw_chars": n,
            "train_chars": train_end,
            "val_chars": val_end - train_end,
            "test_chars": n - val_end,
            "created_at": now_iso(),
        },
    )


def load_text_splits(dataset: str) -> Dict[str, str]:
    ddir = dataset_dir(dataset)
    return {
        "train": (ddir / "train.txt").read_text(encoding="utf-8", errors="replace"),
        "val": (ddir / "val.txt").read_text(encoding="utf-8", errors="replace"),
        "test": (ddir / "test.txt").read_text(encoding="utf-8", errors="replace"),
    }


def load_vocab(dataset: str) -> Vocabulary:
    payload = load_json(artifact_dir(dataset) / "vocab.json")
    return Vocabulary.from_json(payload)


def build_context_target_arrays(ids: np.ndarray, ctx_len: int, bos_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if ids.ndim != 1:
        raise ValueError("ids must be 1D")
    if len(ids) < 2:
        return np.zeros((0, ctx_len), dtype=np.int16), np.zeros((0,), dtype=np.int16)
    padded = np.concatenate([np.full(ctx_len, bos_id, dtype=np.int16), ids[:-1]])
    windows = np.lib.stride_tricks.sliding_window_view(padded, ctx_len)
    contexts = windows.copy().astype(np.int16)
    targets = ids.astype(np.int16)
    return contexts, targets


def h32(text: str) -> int:
    if murmurhash3_32 is not None:
        return int(murmurhash3_32(text, positive=True))
    import hashlib

    h = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little", signed=False)


class FeatureExtractor:
    """Brute-force sparse feature extractor with fixed namespace budgets."""

    def __init__(self, spec: FeatureSpec, vocab: Vocabulary):
        self.spec = spec
        self.vocab = vocab

        self.off_onehot = 0
        self.off_pos_ngram = self.off_onehot + spec.pos_onehot_dim
        self.off_skip = self.off_pos_ngram + spec.pos_ngram_dim
        self.off_misc = self.off_skip + spec.skipgram_dim

    def _onehot_idx(self, pos: int, tok: int) -> int:
        bucket = tok
        if bucket >= self.spec.onehot_vocab_bins:
            bucket = self.spec.onehot_vocab_bins - 1
        return self.off_onehot + pos * self.spec.onehot_vocab_bins + bucket

    def _hash_ns(self, ns: str, key: str, dim: int, offset: int) -> int:
        return offset + (h32(f"{ns}|{key}") % dim)

    def _compress_features(self, context: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = len(context)
        if n == 0:
            return out

        # LZ-ish longest previous match in context suffix.
        tail = context.tolist()
        best_len = 0
        best_dist = 0
        max_check = min(64, n)
        for dist in range(1, max_check + 1):
            l = 0
            while l < n - dist and tail[n - 1 - l] == tail[n - 1 - dist - l] and l < 32:
                l += 1
            if l > best_len:
                best_len = l
                best_dist = dist

        out["lz_match_len"] = float(best_len)
        out["lz_match_dist"] = float(best_dist)

        for w in (16, 64, 256):
            chunk = context[-min(w, n) :]
            counts = np.bincount(chunk, minlength=self.vocab.size).astype(np.float64)
            probs = counts / max(1.0, counts.sum())
            nz = probs[probs > 0]
            entropy = float(-(nz * np.log(nz)).sum())
            out[f"entropy_{w}"] = entropy

        # PPM-ish escape depth within context only.
        escape_depth = 0
        max_order = min(12, n - 1)
        for order in range(max_order, 0, -1):
            suffix = tuple(context[-order:].tolist())
            found = False
            for i in range(0, n - order):
                if tuple(context[i : i + order].tolist()) == suffix:
                    found = True
                    break
            if found:
                break
            escape_depth += 1
        out["ppm_escape_depth"] = float(escape_depth)

        return out

    def _struct_features(self, context: np.ndarray) -> Dict[str, float]:
        ch_ids = context.tolist()
        chars = [self.vocab.id_to_token[int(i)] if 0 <= int(i) < self.vocab.size else "?" for i in ch_ids]
        text = "".join(c if len(c) == 1 else "" for c in chars)

        out: Dict[str, float] = {}
        out["quote_parity"] = float(text.count('"') % 2)
        out["apostrophe_parity"] = float(text.count("'") % 2)
        out["open_paren_balance"] = float(text.count("(") - text.count(")"))
        out["open_bracket_balance"] = float(text.count("[") - text.count("]"))
        out["open_brace_balance"] = float(text.count("{") - text.count("}"))

        trailing_space = 0
        for c in reversed(text):
            if c == " ":
                trailing_space += 1
            else:
                break
        out["trailing_space_run"] = float(trailing_space)

        last_newline = text.rfind("\n")
        out["dist_to_newline"] = float((len(text) - 1 - last_newline) if last_newline >= 0 else len(text))
        return out

    def _count_features(self, context: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        windows = (16, 64, 256)
        classes = {
            "alpha": set(string.ascii_letters),
            "digit": set(string.digits),
            "space": {" ", "\t", "\n"},
            "punct": set(string.punctuation),
        }

        chars = []
        for i in context.tolist():
            tok = self.vocab.id_to_token[int(i)] if 0 <= int(i) < self.vocab.size else "?"
            chars.append(tok if len(tok) == 1 else "?")

        for w in windows:
            chunk = chars[-min(w, len(chars)) :]
            denom = max(1.0, float(len(chunk)))
            for cname, cset in classes.items():
                count = sum(1 for c in chunk if c in cset)
                out[f"{cname}_ratio_{w}"] = count / denom

        # recency of frequent symbols
        tracked = [" ", "\n", "e", "t", "a", ".", ",", "(", ")"]
        for sym in tracked:
            dist = len(chars)
            for j in range(len(chars) - 1, -1, -1):
                if chars[j] == sym:
                    dist = len(chars) - 1 - j
                    break
            out[f"recency_{repr(sym)}"] = float(dist)
        return out

    def extract_indices_values(
        self,
        context: np.ndarray,
        record_raw_keys: bool = False,
    ) -> Tuple[List[int], List[float], Optional[Dict[str, List[str]]]]:
        idx_vals: Dict[int, float] = defaultdict(float)
        raw_keys: Optional[Dict[str, List[str]]] = {"POS_NGRAM": [], "SKIPGRAM": [], "MISC": []} if record_raw_keys else None

        # POS_ONEHOT
        for p, tok in enumerate(context.tolist()):
            idx = self._onehot_idx(p, int(tok))
            idx_vals[idx] += 1.0

        # POS_NGRAM at all offsets (n = 2..5)
        c = context.tolist()
        for n in (2, 3, 4, 5):
            if len(c) < n:
                continue
            for i in range(0, len(c) - n + 1):
                ng = c[i : i + n]
                key = f"i={i}|n={n}|v={','.join(map(str, ng))}"
                idx = self._hash_ns("POS_NGRAM", key, self.spec.pos_ngram_dim, self.off_pos_ngram)
                idx_vals[idx] += 1.0
                if raw_keys is not None:
                    raw_keys["POS_NGRAM"].append(key)

        # SKIPGRAM with gaps 1/2/4/8 over tail 96 positions
        start = max(0, len(c) - 96)
        tail = c[start:]
        for i in range(len(tail)):
            a = tail[i]
            for gap in (1, 2, 4, 8):
                j = i + gap + 1
                if j >= len(tail):
                    continue
                b = tail[j]
                key = f"a={a}|b={b}|gap={gap}|i={i}"
                idx = self._hash_ns("SKIPGRAM", key, self.spec.skipgram_dim, self.off_skip)
                idx_vals[idx] += 1.0
                if raw_keys is not None:
                    raw_keys["SKIPGRAM"].append(key)

        misc_features = {}
        misc_features.update(self._count_features(context))
        misc_features.update(self._struct_features(context))
        misc_features.update(self._compress_features(context))

        for k, v in misc_features.items():
            key = f"{k}={round(float(v), 5)}"
            idx = self._hash_ns("MISC", key, self.spec.misc_dim, self.off_misc)
            idx_vals[idx] += float(v)
            if raw_keys is not None:
                raw_keys["MISC"].append(key)

        indices = list(idx_vals.keys())
        values = [idx_vals[i] for i in indices]
        return indices, values, raw_keys

    def transform(self, contexts: np.ndarray) -> sparse.csr_matrix:
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for r in range(contexts.shape[0]):
            idx, vals, _ = self.extract_indices_values(contexts[r], record_raw_keys=False)
            rows.extend([r] * len(idx))
            cols.extend(idx)
            data.extend(vals)
        mat = sparse.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(contexts.shape[0], self.spec.total_dim))
        mat.sum_duplicates()
        return mat

    def estimate_collisions(self, contexts: np.ndarray, sample_size: int = 100_000) -> Dict[str, Dict[str, float]]:
        rng = np.random.default_rng(1337)
        n = contexts.shape[0]
        if n == 0:
            return {}
        if n > sample_size:
            sample_idx = rng.choice(n, size=sample_size, replace=False)
            sample = contexts[sample_idx]
        else:
            sample = contexts

        uniq_raw: Dict[str, set] = {"POS_NGRAM": set(), "SKIPGRAM": set(), "MISC": set()}
        uniq_bin: Dict[str, set] = {"POS_NGRAM": set(), "SKIPGRAM": set(), "MISC": set()}

        for row in sample:
            _, _, raw = self.extract_indices_values(row, record_raw_keys=True)
            assert raw is not None
            for ns in ("POS_NGRAM", "SKIPGRAM", "MISC"):
                for k in raw[ns]:
                    uniq_raw[ns].add(k)
                    if ns == "POS_NGRAM":
                        idx = self._hash_ns("POS_NGRAM", k, self.spec.pos_ngram_dim, 0)
                    elif ns == "SKIPGRAM":
                        idx = self._hash_ns("SKIPGRAM", k, self.spec.skipgram_dim, 0)
                    else:
                        idx = self._hash_ns("MISC", k, self.spec.misc_dim, 0)
                    uniq_bin[ns].add(idx)

        out: Dict[str, Dict[str, float]] = {}
        for ns in uniq_raw:
            raw_cnt = max(1, len(uniq_raw[ns]))
            bin_cnt = len(uniq_bin[ns])
            collision = float(1.0 - (bin_cnt / raw_cnt))
            out[ns] = {
                "unique_raw": float(raw_cnt),
                "unique_bins": float(bin_cnt),
                "collision_rate": collision,
            }
        return out


class PPMDModel:
    """Simple PPM-D-like model with method-D style escaping."""

    def __init__(self, vocab_size: int, max_order: int = 12):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.tables: List[MutableMapping[Tuple[int, ...], Counter]] = [defaultdict(Counter) for _ in range(max_order + 1)]
        self.suffix_backend: str = "table_only"
        self.suffix_array: Optional[np.ndarray] = None

    def fit(self, ids: np.ndarray) -> None:
        n = len(ids)
        # Optional suffix-array backend for exact suffix infrastructure.
        try:
            import pydivsufsort  # type: ignore

            byte_ids = np.asarray(ids % 256, dtype=np.uint8)
            self.suffix_array = pydivsufsort.divsufsort(byte_ids)
            self.suffix_backend = "pydivsufsort"
        except Exception:
            self.suffix_array = None
            self.suffix_backend = "table_only"

        for i in range(n):
            y = int(ids[i])
            max_o = min(self.max_order, i)
            for o in range(0, max_o + 1):
                if o == 0:
                    ctx: Tuple[int, ...] = ()
                else:
                    ctx = tuple(ids[i - o : i].tolist())
                self.tables[o][ctx][y] += 1

    def predict_distribution(self, context: Sequence[int]) -> Tuple[np.ndarray, Dict[str, float]]:
        probs = np.zeros(self.vocab_size, dtype=np.float64)
        exclusion: set = set()
        remaining = 1.0
        max_depth = 0
        neighbor_count = 0

        c = list(context)
        for o in range(min(self.max_order, len(c)), -1, -1):
            ctx = tuple(c[-o:]) if o > 0 else ()
            counts = self.tables[o].get(ctx)
            if not counts:
                continue
            if o > max_depth:
                max_depth = o
                neighbor_count = int(sum(counts.values()))

            total = float(sum(counts.values()))
            unique = float(len(counts))
            denom = total + unique
            if denom <= 0:
                continue

            for tok, cnt in counts.items():
                if tok in exclusion:
                    continue
                probs[tok] += remaining * (float(cnt) / denom)
            exclusion.update(counts.keys())
            escape = unique / denom
            remaining *= escape
            if remaining < 1e-9:
                break

        if remaining > 0:
            allowed = [i for i in range(self.vocab_size) if i not in exclusion]
            if not allowed:
                allowed = list(range(self.vocab_size))
            add = remaining / float(len(allowed))
            probs[allowed] += add

        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum()

        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        dispersion = float(np.std(probs))
        conf = {
            "max_match_depth": float(max_depth),
            "neighbor_count": float(neighbor_count),
            "entropy": entropy,
            "dispersion": dispersion,
            "suffix_backend": self.suffix_backend,
        }
        return probs.astype(np.float32), conf

    def evaluate_bpc(self, ids: np.ndarray, ctx_len: int) -> Dict[str, float]:
        n = len(ids)
        if n == 0:
            return {"nll": float("nan"), "bpc": float("nan"), "count": 0}
        ll = 0.0
        for i in range(n):
            lo = max(0, i - ctx_len)
            ctx = ids[lo:i]
            dist, _ = self.predict_distribution(ctx)
            ll += -math.log(float(dist[int(ids[i])]))
        nll = ll / n
        bpc = nll / math.log(2)
        return {"nll": float(nll), "bpc": float(bpc), "count": int(n)}


class SimHashIndex:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.bucket_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    @staticmethod
    def _simhash(context: Sequence[int]) -> int:
        vec = np.zeros(64, dtype=np.int32)
        c = list(context)
        for i in range(max(0, len(c) - 64), len(c) - 2):
            key = f"{c[i]}|{c[i+1]}|{c[i+2]}"
            h = h32(key)
            h2 = ((h << 32) | h32(key + "x")) & ((1 << 64) - 1)
            for b in range(64):
                if (h2 >> b) & 1:
                    vec[b] += 1
                else:
                    vec[b] -= 1
        out = 0
        for b in range(64):
            if vec[b] >= 0:
                out |= 1 << b
        return int(out)

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def fit(self, contexts: np.ndarray, targets: np.ndarray, max_entries: int = 500_000) -> None:
        n = contexts.shape[0]
        if n == 0:
            return
        stride = max(1, n // max_entries)
        for i in range(0, n, stride):
            h = self._simhash(contexts[i])
            bucket = h >> 48
            self.bucket_map[bucket].append((h, int(targets[i])))

    def query(self, context: Sequence[int], top_k: int = 256) -> Tuple[np.ndarray, Dict[str, float]]:
        hq = self._simhash(context)
        bucket = hq >> 48
        entries = self.bucket_map.get(bucket, [])
        scores: List[Tuple[float, int]] = []
        for h, tok in entries:
            d = self._hamming(hq, h)
            w = math.exp(-d / 8.0)
            scores.append((w, tok))
        scores.sort(key=lambda x: x[0], reverse=True)
        scores = scores[:top_k]

        probs = np.zeros(self.vocab_size, dtype=np.float64)
        for w, tok in scores:
            probs[tok] += w
        total = probs.sum()
        if total <= 0:
            probs += 1.0
            total = probs.sum()
        probs /= total

        nz = probs[probs > 0]
        conf = {
            "neighbor_count": float(len(scores)),
            "entropy": float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0,
            "dispersion": float(np.std(probs)),
            "max_match_depth": 0.0,
        }
        return probs.astype(np.float32), conf


class OverlapIndex:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.gram_to_counter: Dict[int, Counter] = defaultdict(Counter)

    @staticmethod
    def _grams(context: Sequence[int], n: int = 3, tail: int = 64) -> List[int]:
        c = list(context[-tail:])
        out = []
        if len(c) < n:
            return out
        for i in range(0, len(c) - n + 1):
            key = ",".join(str(x) for x in c[i : i + n])
            out.append(h32(key))
        return out

    def fit(self, contexts: np.ndarray, targets: np.ndarray, max_entries: int = 500_000) -> None:
        n = contexts.shape[0]
        if n == 0:
            return
        stride = max(1, n // max_entries)
        for i in range(0, n, stride):
            grams = self._grams(contexts[i])
            tok = int(targets[i])
            for g in grams:
                self.gram_to_counter[g][tok] += 1

    def query(self, context: Sequence[int]) -> Tuple[np.ndarray, Dict[str, float]]:
        grams = self._grams(context)
        probs = np.zeros(self.vocab_size, dtype=np.float64)
        matched = 0
        for g in grams:
            counter = self.gram_to_counter.get(g)
            if not counter:
                continue
            matched += 1
            for tok, cnt in counter.items():
                probs[tok] += float(cnt)
        total = probs.sum()
        if total <= 0:
            probs += 1.0
            total = probs.sum()
        probs /= total
        nz = probs[probs > 0]
        conf = {
            "neighbor_count": float(matched),
            "entropy": float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0,
            "dispersion": float(np.std(probs)),
            "max_match_depth": 0.0,
        }
        return probs.astype(np.float32), conf


class RetrievalStack:
    def __init__(self, vocab_size: int, ppm_order: int = 12):
        self.vocab_size = vocab_size
        self.ppm = PPMDModel(vocab_size=vocab_size, max_order=ppm_order)
        self.simhash = SimHashIndex(vocab_size=vocab_size)
        self.overlap = OverlapIndex(vocab_size=vocab_size)

    def fit(self, train_ids: np.ndarray, contexts: np.ndarray, targets: np.ndarray) -> None:
        self.ppm.fit(train_ids)
        self.simhash.fit(contexts, targets)
        self.overlap.fit(contexts, targets)

    def query_all(self, context: Sequence[int]) -> Dict[str, Any]:
        d_ppm, c_ppm = self.ppm.predict_distribution(context)
        d_sim, c_sim = self.simhash.query(context)
        d_ov, c_ov = self.overlap.query(context)
        return {
            "dist": {
                "ppm": d_ppm,
                "simhash": d_sim,
                "overlap": d_ov,
            },
            "conf": {
                "ppm": c_ppm,
                "simhash": c_sim,
                "overlap": c_ov,
            },
        }


class NGramBaseline:
    """Simple character 5-gram with add-k smoothing."""

    def __init__(self, vocab_size: int, n: int = 5, k: float = 0.1):
        self.vocab_size = vocab_size
        self.n = n
        self.k = k
        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

    def fit(self, ids: np.ndarray) -> None:
        for i in range(len(ids)):
            lo = max(0, i - (self.n - 1))
            ctx = tuple(ids[lo:i].tolist())
            self.counts[ctx][int(ids[i])] += 1

    def predict(self, context: Sequence[int]) -> np.ndarray:
        c = list(context)
        for o in range(self.n - 1, -1, -1):
            ctx = tuple(c[-o:]) if o > 0 else ()
            counter = self.counts.get(ctx)
            if counter:
                vec = np.full(self.vocab_size, self.k, dtype=np.float64)
                for tok, cnt in counter.items():
                    vec[tok] += float(cnt)
                vec /= vec.sum()
                return vec.astype(np.float32)
        vec = np.ones(self.vocab_size, dtype=np.float32)
        vec /= vec.sum()
        return vec

    def evaluate(self, ids: np.ndarray, ctx_len: int) -> Dict[str, float]:
        n = len(ids)
        ll = 0.0
        for i in range(n):
            lo = max(0, i - ctx_len)
            p = self.predict(ids[lo:i])
            ll += -math.log(float(p[int(ids[i])]))
        nll = ll / max(1, n)
        return {
            "nll": float(nll),
            "bpc": float(nll / math.log(2)),
            "ppl": float(math.exp(nll)),
            "count": int(n),
        }


def _require_catboost() -> Any:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "catboost is required for this command. Install with: pip install catboost"
        ) from e
    return CatBoostClassifier, Pool


class CascadeModel:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.stage_models: Dict[str, Any] = {}
        self.n_classes: Optional[int] = None

    def load(self) -> None:
        CatBoostClassifier, _ = _require_catboost()
        meta_path = self.artifact_path / "cascade_meta.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            self.n_classes = int(meta.get("n_classes", 0)) or None
        for s in ("stage1", "stage2", "stage3"):
            model = CatBoostClassifier()
            model.load_model(str(self.artifact_path / f"cascade_{s}.cbm"))
            self.stage_models[s] = model

    def _predict_proba_full(self, model: Any, X: sparse.csr_matrix) -> np.ndarray:
        raw = np.asarray(model.predict_proba(X), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if self.n_classes is None or raw.shape[1] == self.n_classes:
            return raw

        classes_attr = getattr(model, "classes_", None)
        if classes_attr is None:
            return raw
        classes = np.asarray(classes_attr, dtype=np.int64).reshape(-1)
        full = np.full((raw.shape[0], self.n_classes), 1e-12, dtype=np.float32)
        for j, cls in enumerate(classes.tolist()):
            if 0 <= cls < self.n_classes:
                full[:, cls] = raw[:, j]
        full /= np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
        return full

    def save_meta(self, payload: Mapping[str, Any]) -> None:
        save_json(self.artifact_path / "cascade_meta.json", dict(payload))

    def predict_stage_probs(self, base_row: sparse.csr_matrix) -> Dict[str, np.ndarray]:
        s1 = self._predict_proba_full(self.stage_models["stage1"], base_row)[0]
        x2 = sparse.hstack([base_row, sparse.csr_matrix(s1.reshape(1, -1))], format="csr")
        s2 = self._predict_proba_full(self.stage_models["stage2"], x2)[0]
        delta = s2 - s1
        x3 = sparse.hstack(
            [
                base_row,
                sparse.csr_matrix(s1.reshape(1, -1)),
                sparse.csr_matrix(s2.reshape(1, -1)),
                sparse.csr_matrix(delta.reshape(1, -1)),
            ],
            format="csr",
        )
        s3 = self._predict_proba_full(self.stage_models["stage3"], x3)[0]
        return {
            "stage1": np.asarray(s1, dtype=np.float32),
            "stage2": np.asarray(s2, dtype=np.float32),
            "stage3": np.asarray(s3, dtype=np.float32),
        }


def oof_split_indices(
    n: int,
    k_folds: int,
    seed: int,
    y: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    if y is not None:
        y = np.asarray(y)
        class_counts = np.bincount(y.astype(np.int64))
        nonzero = class_counts[class_counts > 0]
        if len(nonzero) > 0 and int(nonzero.min()) >= k_folds:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            return [(train_idx, val_idx) for train_idx, val_idx in skf.split(idx, y)]
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(idx)]


def fit_catboost_multiclass(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    params: StageParams,
    thread_count: int,
    seed: int,
    classes_count: Optional[int] = None,
) -> Any:
    CatBoostClassifier, Pool = _require_catboost()
    if classes_count is None:
        classes_count = int(max(np.max(y_train), np.max(y_val)) + 1)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        classes_count=classes_count,
        depth=params.depth,
        iterations=params.iterations,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        rsm=params.rsm,
        od_type="Iter",
        od_wait=100,
        random_seed=seed,
        thread_count=thread_count,
        verbose=False,
    )
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    try:
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    except Exception:
        # Small-sample folds can miss classes in train; fallback avoids eval-label mismatch crashes.
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            classes_count=classes_count,
            depth=params.depth,
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            l2_leaf_reg=params.l2_leaf_reg,
            rsm=params.rsm,
            od_type="Iter",
            od_wait=100,
            random_seed=seed,
            thread_count=thread_count,
            verbose=False,
        )
        model.fit(train_pool, use_best_model=False)
    return model


def train_cascade_pipeline(
    dataset: str,
    config: CascadeConfig,
    max_examples: Optional[int] = None,
    iteration_scale: float = 1.0,
) -> None:
    adir = artifact_dir(dataset)
    ensure_dir(adir)

    X = sparse.load_npz(adir / "features_train.npz")
    y = np.load(adir / "targets_train.npy")
    if max_examples is not None and max_examples < X.shape[0]:
        X = X[:max_examples]
        y = y[:max_examples]

    gate_path = adir / "ppm_gate.json"
    gate_scale = 1.0
    if gate_path.exists():
        gate = load_json(gate_path)
        if float(gate.get("bpc", 999)) > 1.52:
            gate_scale = 1.2
    stage_scale = gate_scale * float(iteration_scale)

    def scaled(p: StageParams) -> StageParams:
        return StageParams(
            depth=p.depth,
            iterations=int(p.iterations * stage_scale),
            learning_rate=p.learning_rate,
            l2_leaf_reg=p.l2_leaf_reg,
            rsm=p.rsm,
        )

    stage_params = {
        "stage1": scaled(config.stage1),
        "stage2": scaled(config.stage2),
        "stage3": scaled(config.stage3),
    }

    vocab = load_vocab(dataset)
    n_classes = int(vocab.size)
    splits = oof_split_indices(X.shape[0], config.k_folds, config.random_seed, y=y)

    oof_logits: Dict[str, np.ndarray] = {}
    full_models: Dict[str, Any] = {}

    stage_inputs: Dict[str, sparse.csr_matrix] = {"stage1": X}

    def expand_probs(model: Any, raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == n_classes:
            return arr
        classes_attr = getattr(model, "classes_", None)
        if classes_attr is None:
            return arr
        classes = np.asarray(classes_attr, dtype=np.int64).reshape(-1)
        full = np.full((arr.shape[0], n_classes), 1e-12, dtype=np.float32)
        for j, cls in enumerate(classes.tolist()):
            if 0 <= cls < n_classes:
                full[:, cls] = arr[:, j]
        full /= np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
        return full

    for stage in ("stage1", "stage2", "stage3"):
        Xin = stage_inputs[stage]
        oof = np.zeros((Xin.shape[0], n_classes), dtype=np.float32)

        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            model = fit_catboost_multiclass(
                Xin[tr_idx],
                y[tr_idx],
                Xin[va_idx],
                y[va_idx],
                stage_params[stage],
                thread_count=config.thread_count,
                seed=config.random_seed + fold_id,
                classes_count=n_classes,
            )
            preds = model.predict_proba(Xin[va_idx])
            oof[va_idx] = expand_probs(model, preds)

        oof_logits[stage] = oof

        # full refit
        full_model = fit_catboost_multiclass(
            Xin,
            y,
            Xin[: min(5000, Xin.shape[0])],
            y[: min(5000, Xin.shape[0])],
            stage_params[stage],
            thread_count=config.thread_count,
            seed=config.random_seed + 999,
            classes_count=n_classes,
        )
        full_models[stage] = full_model
        full_model.save_model(str(adir / f"cascade_{stage}.cbm"))

        if stage == "stage1":
            stage_inputs["stage2"] = sparse.hstack([X, sparse.csr_matrix(oof)], format="csr")
        elif stage == "stage2":
            delta = oof_logits["stage2"] - oof_logits["stage1"]
            stage_inputs["stage3"] = sparse.hstack(
                [
                    X,
                    sparse.csr_matrix(oof_logits["stage1"]),
                    sparse.csr_matrix(oof_logits["stage2"]),
                    sparse.csr_matrix(delta),
                ],
                format="csr",
            )

    save_json(
        adir / "cascade_meta.json",
        {
            "k_folds": config.k_folds,
            "stage_scale": stage_scale,
            "n_train": int(X.shape[0]),
            "n_classes": int(n_classes),
            "thread_count": config.thread_count,
            "seed": config.random_seed,
            "created_at": now_iso(),
            "stage_params": {
                k: asdict(v) for k, v in stage_params.items()
            },
        },
    )


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x)
    z = np.exp(x - m)
    s = z.sum()
    if not np.isfinite(s) or s <= 0:
        out = np.ones_like(x, dtype=np.float64)
        out /= out.sum()
        return out.astype(np.float32)
    return (z / s).astype(np.float32)


def top_indices(probs: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probs.shape[0])
    if k <= 0:
        return np.zeros((0,), dtype=np.int32)
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return idx.astype(np.int32)


def build_calibrator_feature_row(
    context: np.ndarray,
    cand: int,
    stage_probs: Dict[str, np.ndarray],
    retr: Dict[str, Any],
) -> List[float]:
    p1 = float(stage_probs["stage1"][cand])
    p2 = float(stage_probs["stage2"][cand])
    p3 = float(stage_probs["stage3"][cand])
    top1 = float(np.max(stage_probs["stage3"]))
    margin = top1 - p3

    ppm = float(retr["dist"]["ppm"][cand])
    simh = float(retr["dist"]["simhash"][cand])
    ov = float(retr["dist"]["overlap"][cand])

    recency = len(context)
    c = context.tolist()
    for i in range(len(c) - 1, -1, -1):
        if int(c[i]) == int(cand):
            recency = len(c) - 1 - i
            break

    conf = retr["conf"]

    return [
        p1,
        p2,
        p3,
        margin,
        ppm,
        simh,
        ov,
        float(cand),
        float(recency),
        float(conf["ppm"].get("max_match_depth", 0.0)),
        float(conf["ppm"].get("neighbor_count", 0.0)),
        float(conf["simhash"].get("neighbor_count", 0.0)),
        float(conf["overlap"].get("neighbor_count", 0.0)),
        float(conf["ppm"].get("entropy", 0.0)),
        float(conf["simhash"].get("entropy", 0.0)),
        float(conf["overlap"].get("entropy", 0.0)),
    ]


def select_candidates(
    base_probs: np.ndarray,
    retr: Dict[str, Any],
    candidate_cap: int,
    calibrator_cap: int,
    include_token: Optional[int] = None,
    ppm_mass_floor: float = 0.92,
) -> np.ndarray:
    ppm = retr["dist"]["ppm"]
    simh = retr["dist"]["simhash"]
    ov = retr["dist"]["overlap"]

    idx_stage = top_indices(base_probs, candidate_cap)
    idx_ppm = top_indices(ppm, candidate_cap)
    idx_sim = top_indices(simh, candidate_cap)
    idx_ov = top_indices(ov, candidate_cap)
    cand = np.unique(np.concatenate([idx_stage, idx_ppm, idx_sim, idx_ov]))
    # Ensure top PPM mass tokens are always present in the candidate set.
    ppm_order = np.argsort(-ppm)
    mass = 0.0
    ppm_keep: List[int] = []
    for tok in ppm_order.tolist():
        ppm_keep.append(tok)
        mass += float(ppm[tok])
        if mass >= ppm_mass_floor or len(ppm_keep) >= candidate_cap:
            break
    cand = np.unique(np.concatenate([cand, np.array(ppm_keep, dtype=np.int32)]))
    if include_token is not None:
        cand = np.unique(np.concatenate([cand, np.array([int(include_token)], dtype=np.int32)]))

    if cand.size == 0:
        return cand

    max_engine = np.maximum.reduce([ppm[cand], simh[cand], ov[cand]])
    pre = 0.7 * np.log(np.clip(base_probs[cand], 1e-12, 1.0)) + 0.3 * np.log(np.clip(max_engine, 1e-12, 1.0))
    if len(cand) > calibrator_cap:
        order = np.argsort(-pre)[:calibrator_cap]
        cand = cand[order]
    if include_token is not None and include_token not in cand:
        cand = np.concatenate([cand, np.array([int(include_token)], dtype=np.int32)])
    if len(cand) > calibrator_cap:
        cand = cand[:calibrator_cap]
    return cand


def compute_aux_gate(retr_conf: Mapping[str, Any], depth_threshold: float, entropy_threshold: float) -> float:
    ppm_conf = retr_conf.get("ppm", {}) if isinstance(retr_conf, Mapping) else {}
    depth = float(ppm_conf.get("max_match_depth", 0.0))
    entropy = float(ppm_conf.get("entropy", 999.0))
    if depth >= depth_threshold and entropy <= entropy_threshold:
        return 0.35
    return 1.0


def combine_distribution(
    stage3: np.ndarray,
    retr: Dict[str, Any],
    calibrator_model: Optional[Any],
    context: np.ndarray,
    stage_probs: Dict[str, np.ndarray],
    fusion: FusionWeights,
    candidate_cap: int = 1024,
    calibrator_cap: int = 192,
    use_stage: str = "stage3",
    include_token: Optional[int] = None,
) -> np.ndarray:
    base_key = use_stage if use_stage in stage_probs else "stage3"
    base_probs = stage_probs[base_key]
    vocab_size = base_probs.shape[0]
    ppm = retr["dist"]["ppm"]
    simh = retr["dist"]["simhash"]
    ov = retr["dist"]["overlap"]
    gate = compute_aux_gate(
        retr.get("conf", {}),
        depth_threshold=fusion.aux_depth_threshold,
        entropy_threshold=fusion.aux_entropy_threshold,
    )
    cand = select_candidates(
        base_probs,
        retr,
        candidate_cap=candidate_cap,
        calibrator_cap=calibrator_cap,
        include_token=include_token,
    )
    if cand.size == 0:
        return base_probs

    if calibrator_model is not None and fusion.calibrator_epsilon != 0.0:
        feats = np.array(
            [build_calibrator_feature_row(context, int(t), stage_probs, retr) for t in cand],
            dtype=np.float32,
        )
        if len(feats) == 0:
            cal = np.zeros((len(cand),), dtype=np.float32)
        else:
            if fusion.calibrator_mode == "delta":
                cal = np.asarray(calibrator_model.predict(feats), dtype=np.float32).reshape(-1)
                cal = np.clip(cal, -fusion.calibrator_clamp, fusion.calibrator_clamp)
            else:
                proba = calibrator_model.predict_proba(feats)[:, 1]
                cal = np.asarray(proba, dtype=np.float32)
    else:
        cal = np.zeros((len(cand),), dtype=np.float32)

    if fusion.fusion_mode == "ppm_primary":
        score = (
            fusion.ppm_alpha * np.log(np.clip(ppm[cand], 1e-12, 1.0))
            + fusion.stage_beta * np.log(np.clip(base_probs[cand], 1e-12, 1.0))
            + gate * fusion.simhash_gamma * np.log(np.clip(simh[cand], 1e-12, 1.0))
            + gate * fusion.overlap_delta * np.log(np.clip(ov[cand], 1e-12, 1.0))
        )
    else:
        score = (
            fusion.stage_beta * np.log(np.clip(base_probs[cand], 1e-12, 1.0))
            + fusion.ppm_alpha * np.log(np.clip(ppm[cand], 1e-12, 1.0))
            + fusion.simhash_gamma * np.log(np.clip(simh[cand], 1e-12, 1.0))
            + fusion.overlap_delta * np.log(np.clip(ov[cand], 1e-12, 1.0))
        )

    if fusion.calibrator_mode == "delta":
        score = score + fusion.calibrator_epsilon * cal
    else:
        score = score + fusion.calibrator_epsilon * cal

    p_cand = safe_softmax(score)
    out = np.zeros(vocab_size, dtype=np.float64)
    prior = ppm if fusion.fusion_mode == "ppm_primary" else base_probs
    cand_prior_mass = float(np.clip(np.sum(prior[cand]), 0.70, 0.995))
    out[cand] = cand_prior_mass * p_cand.astype(np.float64)

    tail = np.ones(vocab_size, dtype=bool)
    tail[cand] = False
    tail_mass = max(1e-8, 1.0 - cand_prior_mass)
    if np.any(tail):
        prior_tail = np.clip(prior[tail], 1e-12, 1.0)
        prior_tail /= prior_tail.sum()
        out[tail] = tail_mass * prior_tail

    out = np.clip(out, 1e-12, None)
    out /= out.sum()
    return out.astype(np.float32)


def tune_fusion_weights(
    contexts: np.ndarray,
    targets: np.ndarray,
    base_X: sparse.csr_matrix,
    cascade: CascadeModel,
    retr: RetrievalStack,
    calibrator: Any,
    max_examples: int = 2000,
    candidate_cap: int = 1024,
    calibrator_cap: int = 192,
    use_stage: str = "stage3",
    fusion_mode: str = "ppm_primary",
    calibrator_mode: str = "delta",
) -> FusionWeights:
    n = min(max_examples, contexts.shape[0])
    if fusion_mode == "ppm_primary":
        grid_ppm = [0.8, 1.0, 1.2]
        grid_stage = [0.2, 0.35, 0.5]
        grid_simh = [0.0, 0.08, 0.15]
        grid_ov = [0.0, 0.08, 0.15]
        grid_cal = [0.0, 0.1, 0.2]
    else:
        grid_ppm = [0.5, 1.0]
        grid_stage = [0.8, 1.0]
        grid_simh = [0.05, 0.15]
        grid_ov = [0.05, 0.15]
        grid_cal = [0.1, 0.2]

    best = FusionWeights(fusion_mode=fusion_mode, calibrator_mode=calibrator_mode)
    best_nll = float("inf")

    cache: List[Dict[str, Any]] = []
    for i in range(n):
        stage_probs = cascade.predict_stage_probs(base_X[i])
        r = retr.query_all(contexts[i])
        base_key = use_stage if use_stage in stage_probs else "stage3"
        base_probs = stage_probs[base_key]
        y = int(targets[i])
        cand = select_candidates(
            base_probs,
            r,
            candidate_cap=candidate_cap,
            calibrator_cap=calibrator_cap,
            include_token=y,
        )
        if cand.size == 0:
            continue

        feats = np.array(
            [build_calibrator_feature_row(contexts[i], int(t), stage_probs, r) for t in cand],
            dtype=np.float32,
        )
        if len(feats):
            if calibrator_mode == "delta":
                cal = np.asarray(calibrator.predict(feats), dtype=np.float32).reshape(-1)
                cal = np.clip(cal, -0.35, 0.35)
            else:
                cal = np.asarray(calibrator.predict_proba(feats)[:, 1], dtype=np.float32)
        else:
            cal = np.zeros((len(cand),), dtype=np.float32)

        gate = compute_aux_gate(r.get("conf", {}), depth_threshold=6.0, entropy_threshold=3.0)
        prior = r["dist"]["ppm"] if fusion_mode == "ppm_primary" else base_probs
        cand_prior_mass = float(np.clip(np.sum(prior[cand]), 0.70, 0.995))
        y_in = np.where(cand == y)[0]
        y_idx = int(y_in[0]) if len(y_in) else -1

        tail_prob_y = None
        if y_idx < 0:
            tail = np.ones(prior.shape[0], dtype=bool)
            tail[cand] = False
            if np.any(tail):
                stage_tail = np.clip(prior[tail], 1e-12, 1.0)
                stage_tail /= stage_tail.sum()
                tail_indices = np.where(tail)[0]
                loc = int(np.where(tail_indices == y)[0][0]) if np.any(tail_indices == y) else None
                if loc is not None:
                    tail_prob_y = float((1.0 - cand_prior_mass) * stage_tail[loc])

        cache.append(
            {
                "y_idx": y_idx,
                "tail_prob_y": tail_prob_y,
                "cand_prior_mass": cand_prior_mass,
                "log_ppm": np.log(np.clip(r["dist"]["ppm"][cand], 1e-12, 1.0)),
                "log_stage": np.log(np.clip(base_probs[cand], 1e-12, 1.0)),
                "log_simh": np.log(np.clip(r["dist"]["simhash"][cand], 1e-12, 1.0)),
                "log_ov": np.log(np.clip(r["dist"]["overlap"][cand], 1e-12, 1.0)),
                "gate": float(gate),
                "cal": np.asarray(cal, dtype=np.float32),
            }
        )

    for ppm_a in grid_ppm:
        for st_b in grid_stage:
            for sim_g in grid_simh:
                for ov_d in grid_ov:
                    for cal_e in grid_cal:
                        w = FusionWeights(
                            ppm_alpha=ppm_a,
                            stage_beta=st_b,
                            simhash_gamma=sim_g,
                            overlap_delta=ov_d,
                            calibrator_epsilon=cal_e,
                            fusion_mode=fusion_mode,
                            calibrator_mode=calibrator_mode,
                        )
                        ll = 0.0
                        for entry in cache:
                            score = w.ppm_alpha * entry["log_ppm"] + w.stage_beta * entry["log_stage"]
                            score += (entry["gate"] * w.simhash_gamma) * entry["log_simh"]
                            score += (entry["gate"] * w.overlap_delta) * entry["log_ov"]
                            score += w.calibrator_epsilon * entry["cal"]
                            log_norm = float(logsumexp(score))
                            y_idx = int(entry["y_idx"])
                            if y_idx >= 0:
                                log_py = math.log(max(1e-12, float(entry["cand_prior_mass"]))) + float(score[y_idx]) - log_norm
                            else:
                                tail_prob = float(entry["tail_prob_y"] or 1e-12)
                                log_py = math.log(max(1e-12, tail_prob))
                            ll += -log_py
                        nll = ll / max(1, len(cache))
                        if nll < best_nll:
                            best_nll = nll
                            best = w
    return best


def _require_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for benchmark transformer baseline. Install with: pip install torch") from e
    return torch, nn, F


class TinyGPTModule:
    def __init__(self, vocab_size: int, n_layer: int, n_embd: int, block_size: int, n_head: int = 4):
        torch, nn, _ = _require_torch()

        class CausalSelfAttention(nn.Module):
            def __init__(self, n_embd: int, n_head: int, block_size: int):
                super().__init__()
                self.n_head = n_head
                self.key = nn.Linear(n_embd, n_embd)
                self.query = nn.Linear(n_embd, n_embd)
                self.value = nn.Linear(n_embd, n_embd)
                self.proj = nn.Linear(n_embd, n_embd)
                self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

            def forward(self, x):
                B, T, C = x.size()
                k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
                att = torch.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.proj(y)

        class Block(nn.Module):
            def __init__(self, n_embd: int, n_head: int, block_size: int):
                super().__init__()
                self.ln1 = nn.LayerNorm(n_embd)
                self.sa = CausalSelfAttention(n_embd, n_head, block_size)
                self.ln2 = nn.LayerNorm(n_embd)
                self.ff = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd),
                )

            def forward(self, x):
                x = x + self.sa(self.ln1(x))
                x = x + self.ff(self.ln2(x))
                return x

        class TinyGPT(nn.Module):
            def __init__(self, vocab_size: int, n_layer: int, n_embd: int, block_size: int, n_head: int):
                super().__init__()
                self.block_size = block_size
                self.token_emb = nn.Embedding(vocab_size, n_embd)
                self.pos_emb = nn.Embedding(block_size, n_embd)
                self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(n_embd)
                self.head = nn.Linear(n_embd, vocab_size)

            def forward(self, idx, targets=None):
                B, T = idx.size()
                pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
                x = self.token_emb(idx) + self.pos_emb(pos)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)
                loss = None
                if targets is not None:
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss

        self.torch = torch
        self.nn = nn
        self.model = TinyGPT(vocab_size, n_layer, n_embd, block_size, n_head)


def run_tiny_transformer_budget(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_embd: int,
    budget_minutes: int,
    seed: int,
    thread_count: int,
) -> Dict[str, float]:
    torch, _, _ = _require_torch()
    set_global_thread_env(thread_count)
    torch.set_num_threads(thread_count)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_wrap = TinyGPTModule(vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, block_size=block_size)
    model = model_wrap.model
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    device = torch.device("cpu")
    model.to(device)

    def get_batch(arr: np.ndarray, batch_size: int = 32) -> Tuple[Any, Any]:
        ix = np.random.randint(0, len(arr) - block_size - 1, size=(batch_size,))
        x = np.stack([arr[i : i + block_size] for i in ix])
        y = np.stack([arr[i + 1 : i + block_size + 1] for i in ix])
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

    end_time = time.time() + budget_minutes * 60
    steps = 0
    while time.time() < end_time:
        xb, yb = get_batch(train_ids)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        steps += 1

    model.eval()

    def eval_nll(arr: np.ndarray, max_batches: int = 128) -> float:
        losses = []
        with torch.no_grad():
            for _ in range(max_batches):
                xb, yb = get_batch(arr)
                _, loss = model(xb, yb)
                losses.append(float(loss.item()))
        return float(np.mean(losses))

    nll = eval_nll(test_ids)
    return {
        "nll": nll,
        "bpc": float(nll / math.log(2)),
        "ppl": float(math.exp(nll)),
        "steps": int(steps),
        "budget_minutes": int(budget_minutes),
        "n_layer": int(n_layer),
        "n_embd": int(n_embd),
        "seed": int(seed),
    }


def load_feature_spec(dataset: str) -> FeatureSpec:
    path = artifact_dir(dataset) / "feature_spec.json"
    if path.exists():
        payload = load_json(path)
        allowed = set(FeatureSpec.__dataclass_fields__.keys())
        filtered = {k: v for k, v in payload.items() if k in allowed}
        spec = FeatureSpec(**filtered)
        spec.recompute_total()
        return spec
    spec = FeatureSpec()
    spec.recompute_total()
    return spec


def save_feature_spec(dataset: str, spec: FeatureSpec, collisions: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    payload = asdict(spec)
    if collisions is not None:
        payload["collision_estimates"] = collisions
    save_json(artifact_dir(dataset) / "feature_spec.json", payload)


def cmd_prepare(args: argparse.Namespace) -> None:
    record_run_step(args.dataset, "prepare", args)
    prepare_dataset(args.dataset)
    print(f"Prepared dataset: {args.dataset}")


def cmd_ppm_gate(args: argparse.Namespace) -> None:
    dataset = args.dataset
    record_run_step(dataset, "ppm-gate", args)
    if dataset != "tinyshakespeare":
        raise ValueError("ppm-gate is defined for tinyshakespeare in this spec")

    vocab = load_vocab(dataset)
    splits = load_text_splits(dataset)
    train_ids = vocab.encode(splits["train"])
    test_ids = vocab.encode(splits["test"])
    if args.max_train_tokens and len(train_ids) > args.max_train_tokens:
        train_ids = train_ids[: args.max_train_tokens]
    if args.max_eval_tokens and len(test_ids) > args.max_eval_tokens:
        test_ids = test_ids[: args.max_eval_tokens]

    ppm = PPMDModel(vocab_size=vocab.size, max_order=12)
    t0 = time.time()
    ppm.fit(train_ids)
    metrics = ppm.evaluate_bpc(test_ids, ctx_len=256)
    dur = time.time() - t0

    decision = "keep_retrieval_heavy" if metrics["bpc"] <= 1.52 else "increase_parametric_capacity"

    adir = artifact_dir(dataset)
    ensure_dir(adir)
    joblib.dump(ppm, adir / "ppm_model.pkl")
    out = {
        "dataset": dataset,
        "max_order": 12,
        "suffix_backend": ppm.suffix_backend,
        "train_tokens": int(len(train_ids)),
        "eval_tokens": int(len(test_ids)),
        "nll": metrics["nll"],
        "bpc": metrics["bpc"],
        "decision": decision,
        "train_seconds": dur,
        "created_at": now_iso(),
    }
    save_json(adir / "ppm_gate.json", out)
    print(json.dumps(out, indent=2))


def cmd_build_features(args: argparse.Namespace) -> None:
    dataset = args.dataset
    record_run_step(dataset, "build-features", args)
    vocab = load_vocab(dataset)
    splits = load_text_splits(dataset)

    spec = load_feature_spec(dataset)
    extractor = FeatureExtractor(spec, vocab)

    adir = artifact_dir(dataset)
    ensure_dir(adir)

    for split in ("train", "val", "test"):
        ids = vocab.encode(splits[split])
        contexts, targets = build_context_target_arrays(ids, ctx_len=spec.ctx_len, bos_id=vocab.bos_id)
        if args.max_examples and contexts.shape[0] > args.max_examples:
            contexts = contexts[: args.max_examples]
            targets = targets[: args.max_examples]

        np.save(adir / f"contexts_{split}.npy", contexts)
        np.save(adir / f"targets_{split}.npy", targets)

        if split == "train":
            collisions = extractor.estimate_collisions(contexts, sample_size=100_000)
            adjusted = False
            for ns, stats in collisions.items():
                if stats["collision_rate"] > spec.hash_collision_threshold:
                    adjusted = True
                    spec.adjusted_namespaces[ns] = True
                    if ns == "POS_NGRAM":
                        spec.pos_ngram_dim *= 2
                    elif ns == "SKIPGRAM":
                        spec.skipgram_dim *= 2
                    elif ns == "MISC":
                        spec.misc_dim *= 2
            if adjusted:
                spec.recompute_total()
                extractor = FeatureExtractor(spec, vocab)
            save_feature_spec(dataset, spec, collisions=collisions)

        X = extractor.transform(contexts)
        sparse.save_npz(adir / f"features_{split}.npz", X)

    save_json(
        adir / "feature_build_meta.json",
        {
            "dataset": dataset,
            "ctx_len": spec.ctx_len,
            "total_dim": spec.total_dim,
            "created_at": now_iso(),
        },
    )
    print(f"Built features for {dataset} with total_dim={spec.total_dim}")


def cmd_train_cascade(args: argparse.Namespace) -> None:
    record_run_step(args.dataset, "train-cascade", args)
    cfg = CascadeConfig(
        k_folds=3,
        thread_count=args.thread_count or default_thread_count(),
        random_seed=args.seed,
    )
    set_global_thread_env(cfg.thread_count)
    train_cascade_pipeline(
        args.dataset,
        cfg,
        max_examples=args.max_examples,
        iteration_scale=args.iteration_scale,
    )
    print(f"Trained cascade for {args.dataset}")


def cmd_build_retrieval(args: argparse.Namespace) -> None:
    dataset = args.dataset
    record_run_step(dataset, "build-retrieval", args)
    adir = artifact_dir(dataset)
    vocab = load_vocab(dataset)

    splits = load_text_splits(dataset)
    train_ids = vocab.encode(splits["train"])
    if args.max_train_tokens and len(train_ids) > args.max_train_tokens:
        train_ids = train_ids[: args.max_train_tokens]
    contexts = np.load(adir / "contexts_train.npy")
    targets = np.load(adir / "targets_train.npy")
    if args.max_examples and contexts.shape[0] > args.max_examples:
        contexts = contexts[: args.max_examples]
        targets = targets[: args.max_examples]

    stack = RetrievalStack(vocab_size=vocab.size, ppm_order=12)
    t0 = time.time()
    stack.fit(train_ids, contexts, targets)
    dur = time.time() - t0

    joblib.dump(stack, adir / "retrieval_stack.pkl")
    save_json(
        adir / "retrieval_meta.json",
        {
            "dataset": dataset,
            "train_contexts": int(contexts.shape[0]),
            "train_seconds": dur,
            "created_at": now_iso(),
        },
    )
    print(f"Built retrieval stack for {dataset}")


def _load_pipeline(dataset: str) -> Tuple[Vocabulary, FeatureExtractor, CascadeModel, RetrievalStack, Any, FusionWeights]:
    adir = artifact_dir(dataset)
    vocab = load_vocab(dataset)
    spec = load_feature_spec(dataset)
    extractor = FeatureExtractor(spec, vocab)

    cascade = CascadeModel(adir)
    cascade.load()

    retr: RetrievalStack = joblib.load(adir / "retrieval_stack.pkl")

    calibrator_meta = load_json(adir / "calibrator_meta.json") if (adir / "calibrator_meta.json").exists() else {}
    calibrator_mode = calibrator_meta.get("calibrator_mode", "full")
    if calibrator_mode == "delta":
        try:
            from catboost import CatBoostRegressor
        except Exception as e:  # pragma: no cover
            raise RuntimeError("catboost is required for this command. Install with: pip install catboost") from e
        calibrator = CatBoostRegressor()
        calibrator.load_model(str(adir / "calibrator.cbm"))
    else:
        CatBoostClassifier, _ = _require_catboost()
        calibrator = CatBoostClassifier()
        calibrator.load_model(str(adir / "calibrator.cbm"))

    wpath = adir / "fusion_weights.json"
    if wpath.exists():
        fusion = FusionWeights(**normalize_fusion_payload(load_json(wpath)))
    else:
        fusion = FusionWeights()

    return vocab, extractor, cascade, retr, calibrator, fusion


def cmd_train_calibrator(args: argparse.Namespace) -> None:
    dataset = args.dataset
    adir = artifact_dir(dataset)
    record_run_step(dataset, "train-calibrator", args)

    vocab = load_vocab(dataset)

    X_val = sparse.load_npz(adir / "features_val.npz")
    contexts_val = np.load(adir / "contexts_val.npy")
    targets_val = np.load(adir / "targets_val.npy")
    if args.max_examples and X_val.shape[0] > args.max_examples:
        X_val = X_val[: args.max_examples]
        contexts_val = contexts_val[: args.max_examples]
        targets_val = targets_val[: args.max_examples]

    cascade = CascadeModel(adir)
    cascade.load()

    retr: RetrievalStack = joblib.load(adir / "retrieval_stack.pkl")

    feats: List[List[float]] = []
    labels: List[float] = []

    candidate_cap = int(args.candidate_pool)
    calibrator_cap = int(args.calibrator_cap)
    calibrator_mode = str(args.calibrator_mode)
    fusion_mode = str(args.fusion_mode)

    n = X_val.shape[0]
    half = max(1, n // 2)

    for i in range(half):
        stage_probs = cascade.predict_stage_probs(X_val[i])
        r = retr.query_all(contexts_val[i])
        stage3 = stage_probs["stage3"]
        y_true = int(targets_val[i])
        cand = select_candidates(
            stage3,
            r,
            candidate_cap=candidate_cap,
            calibrator_cap=calibrator_cap,
            include_token=y_true,
        )

        # true + up to 20 hard negatives
        cand_order = np.argsort(-stage3[cand])
        hard = [int(cand[j]) for j in cand_order if int(cand[j]) != y_true][:20]
        selected = [y_true] + hard

        for tok in selected:
            feats.append(build_calibrator_feature_row(contexts_val[i], tok, stage_probs, r))
            if calibrator_mode == "delta":
                ppm_tok = float(r["dist"]["ppm"][tok])
                labels.append((1.0 if tok == y_true else 0.0) - ppm_tok)
            else:
                labels.append(1.0 if tok == y_true else 0.0)

    Xc = np.array(feats, dtype=np.float32)
    yc = np.array(labels, dtype=np.float32)

    if calibrator_mode == "delta":
        try:
            from catboost import CatBoostRegressor, Pool
        except Exception as e:  # pragma: no cover
            raise RuntimeError("catboost is required for this command. Install with: pip install catboost") from e
        model: Any = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            depth=6,
            iterations=600,
            learning_rate=0.05,
            l2_leaf_reg=8.0,
            random_seed=args.seed,
            thread_count=args.thread_count or default_thread_count(),
            verbose=False,
        )
        model.fit(Pool(Xc, yc), use_best_model=False)
    else:
        CatBoostClassifier, Pool = _require_catboost()
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            depth=6,
            iterations=600,
            learning_rate=0.05,
            l2_leaf_reg=8.0,
            random_seed=args.seed,
            thread_count=args.thread_count or default_thread_count(),
            verbose=False,
        )
        model.fit(Pool(Xc, yc.astype(np.int32)), use_best_model=False)
    model.save_model(str(adir / "calibrator.cbm"))

    fusion = tune_fusion_weights(
        contexts=contexts_val[half:],
        targets=targets_val[half:],
        base_X=X_val[half:],
        cascade=cascade,
        retr=retr,
        calibrator=model,
        max_examples=min(int(args.tune_max_examples), max(1, n - half)),
        candidate_cap=candidate_cap,
        calibrator_cap=calibrator_cap,
        use_stage="stage3",
        fusion_mode=fusion_mode,
        calibrator_mode=calibrator_mode,
    )
    fusion.ppm_alpha = float(args.ppm_alpha)
    fusion.stage_beta = float(args.stage_beta)
    fusion.simhash_gamma = float(args.simhash_gamma)
    fusion.overlap_delta = float(args.overlap_delta)
    fusion.calibrator_epsilon = float(args.calibrator_epsilon)
    fusion.fusion_mode = fusion_mode
    fusion.calibrator_mode = calibrator_mode
    fusion.calibrator_clamp = float(args.calibrator_clamp)
    fusion.aux_depth_threshold = float(args.aux_depth_threshold)
    fusion.aux_entropy_threshold = float(args.aux_entropy_threshold)
    save_json(adir / "fusion_weights.json", asdict(fusion))

    save_json(
        adir / "calibrator_meta.json",
        {
            "dataset": dataset,
            "train_rows": int(Xc.shape[0]),
            "feature_dim": int(Xc.shape[1]),
            "candidate_pool": candidate_cap,
            "calibrator_cap": calibrator_cap,
            "tune_max_examples": int(args.tune_max_examples),
            "fusion_mode": fusion_mode,
            "calibrator_mode": calibrator_mode,
            "seed": args.seed,
            "thread_count": args.thread_count or default_thread_count(),
            "created_at": now_iso(),
        },
    )

    print(f"Trained calibrator for {dataset}; tuned fusion={asdict(fusion)}")


def evaluate_model(
    dataset: str,
    split: str,
    max_eval_examples: Optional[int] = None,
    fusion_override: Optional[FusionWeights] = None,
    disable_calibrator: bool = False,
    candidate_cap: int = 1024,
    calibrator_cap: int = 192,
    use_stage: str = "stage3",
) -> Dict[str, Any]:
    adir = artifact_dir(dataset)
    vocab, extractor, cascade, retr, calibrator, fusion = _load_pipeline(dataset)
    if fusion_override is not None:
        fusion = fusion_override

    X = sparse.load_npz(adir / f"features_{split}.npz")
    contexts = np.load(adir / f"contexts_{split}.npy")
    targets = np.load(adir / f"targets_{split}.npy")

    if max_eval_examples and X.shape[0] > max_eval_examples:
        X = X[:max_eval_examples]
        contexts = contexts[:max_eval_examples]
        targets = targets[:max_eval_examples]

    ll_stage1 = 0.0
    ll_stage2 = 0.0
    ll_stage3 = 0.0
    ll_final = 0.0
    ll_ppm = 0.0
    top1 = 0

    for i in range(X.shape[0]):
        stage_probs = cascade.predict_stage_probs(X[i])
        y = int(targets[i])

        p1 = float(np.clip(stage_probs["stage1"][y], 1e-12, 1.0))
        p2 = float(np.clip(stage_probs["stage2"][y], 1e-12, 1.0))
        p3 = float(np.clip(stage_probs["stage3"][y], 1e-12, 1.0))
        ll_stage1 += -math.log(p1)
        ll_stage2 += -math.log(p2)
        ll_stage3 += -math.log(p3)

        r = retr.query_all(contexts[i])
        ll_ppm += -math.log(float(np.clip(r["dist"]["ppm"][y], 1e-12, 1.0)))
        pf = combine_distribution(
            stage_probs["stage3"],
            r,
            None if disable_calibrator else calibrator,
            contexts[i],
            stage_probs,
            fusion=fusion,
            candidate_cap=candidate_cap,
            calibrator_cap=calibrator_cap,
            use_stage=use_stage,
            include_token=y,
        )

        ll_final += -math.log(float(np.clip(pf[y], 1e-12, 1.0)))
        if int(np.argmax(pf)) == y:
            top1 += 1

    n = max(1, X.shape[0])
    out = {
        "dataset": dataset,
        "split": split,
        "count": int(n),
        "stage1_nll": float(ll_stage1 / n),
        "stage2_nll": float(ll_stage2 / n),
        "stage3_nll": float(ll_stage3 / n),
        "ppm_nll": float(ll_ppm / n),
        "final_nll": float(ll_final / n),
        "final_bpc": float((ll_final / n) / math.log(2)),
        "final_ppl": float(math.exp(ll_final / n)),
        "top1_acc": float(top1 / n),
        "thread_count": default_thread_count(),
        "created_at": now_iso(),
    }
    return out


def cmd_eval(args: argparse.Namespace) -> None:
    record_run_step(args.dataset, "eval", args)
    out = evaluate_model(
        args.dataset,
        args.split,
        max_eval_examples=args.max_eval_examples,
        candidate_cap=args.candidate_pool,
        calibrator_cap=args.calibrator_cap,
        use_stage=args.use_stage,
    )
    adir = artifact_dir(args.dataset)
    save_json(adir / f"metrics_{args.split}.json", out)

    # N-gram baseline report
    vocab = load_vocab(args.dataset)
    splits = load_text_splits(args.dataset)
    ng = NGramBaseline(vocab_size=vocab.size, n=5, k=0.1)
    ng.fit(vocab.encode(splits["train"]))
    ngm = ng.evaluate(vocab.encode(splits[args.split]), ctx_len=256)
    joblib.dump(ng, adir / "ngram_baseline.pkl")

    out["ngram_nll"] = ngm["nll"]
    out["ngram_bpc"] = ngm["bpc"]
    out["ngram_ppl"] = ngm["ppl"]

    save_json(adir / "metrics.json", out)
    print(json.dumps(out, indent=2))


def apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)
    keep = cdf <= top_p
    if not np.any(keep):
        keep[0] = True
    cutoff = np.where(keep)[0][-1]
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[: cutoff + 1]] = True
    out = np.where(mask, probs, 0.0)
    total = out.sum()
    if total <= 0:
        return probs
    return out / total


def cmd_sample(args: argparse.Namespace) -> None:
    record_run_step(args.dataset, "sample", args)
    vocab, extractor, cascade, retr, calibrator, fusion = _load_pipeline(args.dataset)
    spec = load_feature_spec(args.dataset)
    fusion = fusion_from_args(args, fallback=fusion)

    rng = np.random.default_rng(args.seed)

    prompt_ids = vocab.encode(args.prompt)
    generated = list(prompt_ids.tolist())

    for _ in range(args.tokens):
        ctx = np.full(spec.ctx_len, vocab.bos_id, dtype=np.int16)
        tail = generated[-spec.ctx_len :]
        if tail:
            ctx[-len(tail) :] = np.array(tail, dtype=np.int16)

        X = extractor.transform(ctx.reshape(1, -1))
        stage_probs = cascade.predict_stage_probs(X[0])
        r = retr.query_all(ctx)
        p = combine_distribution(
            stage_probs["stage3"],
            r,
            calibrator,
            ctx,
            stage_probs,
            fusion=fusion,
            candidate_cap=args.candidate_pool,
            calibrator_cap=args.calibrator_cap,
            use_stage=args.use_stage,
        )

        lp = np.log(np.clip(p, 1e-12, 1.0)) / max(1e-6, args.temperature)
        ptemp = safe_softmax(lp)
        ptop = apply_top_p(ptemp, args.top_p)
        nxt = int(rng.choice(np.arange(vocab.size), p=ptop))
        generated.append(nxt)

    text = vocab.decode(generated)
    adir = artifact_dir(args.dataset)
    ensure_dir(adir)
    with (adir / "samples.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n---\nprompt={args.prompt!r} seed={args.seed}\n{text}\n")

    print(text)


def evaluate_model_with_overrides(
    dataset: str,
    split: str,
    max_eval_examples: Optional[int] = None,
    candidate_cap: int = 1024,
    calibrator_cap: int = 192,
    use_stage: str = "stage3",
    fusion_override: Optional[FusionWeights] = None,
    disable_calibrator: bool = False,
) -> Dict[str, Any]:
    out = evaluate_model(
        dataset=dataset,
        split=split,
        max_eval_examples=max_eval_examples,
        fusion_override=fusion_override,
        disable_calibrator=disable_calibrator,
        candidate_cap=candidate_cap,
        calibrator_cap=calibrator_cap,
        use_stage=use_stage,
    )
    out["base_stage"] = use_stage
    if use_stage == "stage1":
        out["base_nll"] = float(out["stage1_nll"])
    elif use_stage == "stage2":
        out["base_nll"] = float(out["stage2_nll"])
    else:
        out["base_nll"] = float(out["stage3_nll"])
    out["candidate_cap"] = int(candidate_cap)
    out["calibrator_cap"] = int(calibrator_cap)
    out["calibrator_enabled"] = bool(not disable_calibrator)
    return out


def cmd_ablate(args: argparse.Namespace) -> None:
    dataset = args.dataset
    split = args.split
    record_run_step(dataset, "ablate", args)
    adir = artifact_dir(dataset)
    ensure_dir(adir)

    if args.config and Path(args.config).exists():
        cfg = load_json(Path(args.config))
        names = list(cfg.get("ablations", []))
    else:
        names = [
            "with_calibrator",
            "no_calibrator",
            "retrieval_ppm_only",
            "retrieval_simhash_only",
            "retrieval_overlap_only",
            "retrieval_all",
            "cascade_depth_1",
            "cascade_depth_2",
            "cascade_depth_3",
            "candidate_k_64",
            "candidate_k_256",
            "candidate_k_512",
            "candidate_k_1024",
        ]

    if (adir / "fusion_weights.json").exists():
        base_fusion = FusionWeights(**normalize_fusion_payload(load_json(adir / "fusion_weights.json")))
    else:
        base_fusion = FusionWeights()
    base_fusion = fusion_from_args(args, fallback=base_fusion)
    results: List[Dict[str, Any]] = []
    retrain_required = bool(args.retrain_required)
    spec = load_feature_spec(dataset)

    def _backup_and_mask_features(mask_kind: str) -> Tuple[Path, str]:
        bdir = adir / "_ablation_backup"
        if bdir.exists():
            shutil.rmtree(bdir)
        ensure_dir(bdir)
        files = [
            "features_train.npz",
            "features_val.npz",
            "features_test.npz",
            "cascade_stage1.cbm",
            "cascade_stage2.cbm",
            "cascade_stage3.cbm",
            "cascade_meta.json",
            "calibrator.cbm",
            "calibrator_meta.json",
            "fusion_weights.json",
        ]
        for name in files:
            src = adir / name
            if src.exists():
                shutil.copy2(src, bdir / name)

        note = ""
        start_onehot = 0
        end_onehot = spec.pos_onehot_dim
        end_pos = end_onehot + spec.pos_ngram_dim
        end_skip = end_pos + spec.skipgram_dim
        end_misc = end_skip + spec.misc_dim
        for split_name in ("train", "val", "test"):
            p = adir / f"features_{split_name}.npz"
            X = sparse.load_npz(p).tocsr(copy=True)
            if mask_kind == "remove_bruteforce_feature_expansion":
                X[:, start_onehot:end_skip] = 0.0
                note = "dropped POS_ONEHOT/POS_NGRAM/SKIPGRAM blocks"
            elif mask_kind == "remove_compression_features":
                # Compression features share MISC hash namespace with count/struct features in v3/v4 extractor.
                X[:, end_skip:end_misc] = 0.0
                note = "approximation: dropped full MISC block (includes compression + count/struct)"
            sparse.save_npz(p, X)
        return bdir, note

    def _restore_backup(bdir: Path) -> None:
        if not bdir.exists():
            return
        for item in bdir.iterdir():
            shutil.copy2(item, adir / item.name)
        shutil.rmtree(bdir, ignore_errors=True)

    for name in names:
        fusion = FusionWeights(**asdict(base_fusion))
        use_stage = "stage3"
        candidate_cap = int(args.candidate_pool)
        calibrator_cap = int(args.calibrator_cap)
        disable_calibrator = False
        supported = True
        reason = ""
        retrain_done = False
        backup_dir: Optional[Path] = None
        retrain_note = ""

        if name == "with_calibrator":
            pass
        elif name == "no_calibrator":
            disable_calibrator = True
            fusion.calibrator_epsilon = 0.0
        elif name == "retrieval_ppm_only":
            fusion.simhash_gamma = 0.0
            fusion.overlap_delta = 0.0
        elif name == "retrieval_simhash_only":
            fusion.ppm_alpha = 0.0
            fusion.overlap_delta = 0.0
        elif name == "retrieval_overlap_only":
            fusion.ppm_alpha = 0.0
            fusion.simhash_gamma = 0.0
        elif name == "retrieval_all":
            pass
        elif name == "cascade_depth_1":
            use_stage = "stage1"
        elif name == "cascade_depth_2":
            use_stage = "stage2"
        elif name == "cascade_depth_3":
            use_stage = "stage3"
        elif name == "candidate_k_64":
            candidate_cap = 64
        elif name == "candidate_k_256":
            candidate_cap = 256
        elif name == "candidate_k_512":
            candidate_cap = 512
        elif name == "candidate_k_1024":
            candidate_cap = 1024
        elif name in {"remove_bruteforce_feature_expansion", "remove_compression_features"}:
            if not retrain_required:
                supported = False
                reason = "requires --retrain-required for train-time feature ablation"
        else:
            supported = False
            reason = "requires retraining or unsupported in inference-only ablation harness"

        if not supported:
            results.append({"ablation": name, "status": "skipped", "reason": reason})
            continue

        try:
            if retrain_required and name in {
                "remove_bruteforce_feature_expansion",
                "remove_compression_features",
                "cascade_depth_1",
                "cascade_depth_2",
                "cascade_depth_3",
                "retrieval_ppm_only",
                "retrieval_simhash_only",
                "retrieval_overlap_only",
                "retrieval_all",
                "with_calibrator",
                "no_calibrator",
            }:
                if name in {"remove_bruteforce_feature_expansion", "remove_compression_features"}:
                    backup_dir, retrain_note = _backup_and_mask_features(name)

                cmd_train_cascade(
                    argparse.Namespace(
                        dataset=dataset,
                        seed=int(args.seed),
                        thread_count=args.thread_count,
                        max_examples=args.max_examples,
                        iteration_scale=args.iteration_scale,
                    )
                )
                cmd_train_calibrator(
                    argparse.Namespace(
                        dataset=dataset,
                        seed=int(args.seed),
                        thread_count=args.thread_count,
                        max_examples=args.max_examples,
                        candidate_pool=candidate_cap,
                        calibrator_cap=calibrator_cap,
                        tune_max_examples=args.tune_max_examples,
                        fusion_mode=fusion.fusion_mode,
                        calibrator_mode=fusion.calibrator_mode,
                        ppm_alpha=fusion.ppm_alpha,
                        stage_beta=fusion.stage_beta,
                        simhash_gamma=fusion.simhash_gamma,
                        overlap_delta=fusion.overlap_delta,
                        calibrator_epsilon=fusion.calibrator_epsilon,
                        calibrator_clamp=fusion.calibrator_clamp,
                        aux_depth_threshold=fusion.aux_depth_threshold,
                        aux_entropy_threshold=fusion.aux_entropy_threshold,
                    )
                )
                retrain_done = True

            metrics = evaluate_model_with_overrides(
                dataset=dataset,
                split=split,
                max_eval_examples=args.max_eval_examples,
                candidate_cap=candidate_cap,
                calibrator_cap=calibrator_cap,
                use_stage=use_stage,
                fusion_override=fusion,
                disable_calibrator=disable_calibrator,
            )
            metrics["ablation"] = name
            metrics["status"] = "ok"
            metrics["retrain_required"] = retrain_required
            metrics["retrain_done"] = retrain_done
            if retrain_note:
                metrics["note"] = retrain_note
            results.append(metrics)
        finally:
            if backup_dir is not None:
                _restore_backup(backup_dir)

    payload = {
        "dataset": dataset,
        "split": split,
        "max_eval_examples": args.max_eval_examples,
        "created_at": now_iso(),
        "results": results,
    }
    save_json(adir / "ablations_results.json", payload)
    print(json.dumps(payload, indent=2))


def cmd_check_acceptance(args: argparse.Namespace) -> None:
    dataset = args.dataset
    record_run_step(dataset, "check-acceptance", args)
    adir = artifact_dir(dataset)
    metrics_test_path = adir / "metrics_test.json"
    metrics_path = metrics_test_path if metrics_test_path.exists() else adir / "metrics.json"
    metrics_val_path = adir / "metrics_val.json"
    if args.benchmark_path and args.benchmark_path != "__none__":
        bench_path = Path(args.benchmark_path)
    elif args.benchmark_path == "__none__":
        bench_path = adir / "__none__"
    else:
        bench_path = adir / "benchmark_crossover_cpu.json"
    ablation_path = adir / "ablations_results.json"
    claim_table_path = adir / "claim_table.json"
    manifest_path = adir / "run_manifest.json"

    report: Dict[str, Any] = {
        "dataset": dataset,
        "created_at": now_iso(),
        "primary": {},
        "secondary": {},
        "sources": {
            "metrics": str(metrics_path) if metrics_path.exists() else None,
            "metrics_val": str(metrics_val_path) if metrics_val_path.exists() else None,
            "benchmark": str(bench_path) if bench_path.exists() else None,
            "ablations": str(ablation_path) if ablation_path.exists() else None,
            "claim_table": str(claim_table_path) if claim_table_path.exists() else None,
            "manifest": str(manifest_path) if manifest_path.exists() else None,
        },
    }

    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    metrics_val = load_json(metrics_val_path) if metrics_val_path.exists() else {}
    bench = load_json(bench_path) if bench_path.exists() else {}
    ablations = load_json(ablation_path) if ablation_path.exists() else {}
    claim_table = load_json(claim_table_path) if claim_table_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    expected_budgets = parse_csv_ints(args.expected_budgets) if str(args.expected_budgets).strip() else []
    expected_seeds = parse_csv_ints(args.expected_seeds) if str(args.expected_seeds).strip() else []
    if not expected_budgets or not expected_seeds:
        exp = manifest.get("benchmark_expectations", {})
        expected_budgets = expected_budgets or [int(x) for x in exp.get("budgets", [])]
        expected_seeds = expected_seeds or [int(x) for x in exp.get("seeds", [])]
    report["expected"] = {"budgets": expected_budgets, "seeds": expected_seeds}

    # Primary 1: >=8% relative NLL improvement vs strongest baseline.
    final_nll = metrics.get("final_nll")
    baselines = [x for x in [metrics.get("ngram_nll"), metrics.get("stage3_nll")] if x is not None]
    if final_nll is not None and baselines:
        strongest = min(float(x) for x in baselines)
        rel_gain = (strongest - float(final_nll)) / max(1e-12, strongest)
        report["primary"]["gate_1_strongest_baseline_rel_gain_ge_8pct"] = {
            "status": "pass" if rel_gain >= 0.08 else "fail",
            "relative_gain": rel_gain,
            "threshold": 0.08,
            "strongest_baseline_nll": strongest,
            "final_nll": float(final_nll),
        }
    else:
        report["primary"]["gate_1_strongest_baseline_rel_gain_ge_8pct"] = {"status": "unknown"}

    # Primary 2/3: crossover and 2/3 seed robustness.
    rr_bpc = None
    stale_benchmark = False
    stale_reason = ""
    if isinstance(bench, dict):
        rr_bpc = bench.get("rr_treelm", {}).get("final_bpc")
    tf_rows = []
    if isinstance(bench, dict):
        for row in bench.get("tiny_transformer", []):
            if row.get("name") == "TF-1L-64d" and row.get("bpc") is not None:
                tf_rows.append(row)

    if tf_rows and expected_budgets and expected_seeds:
        seen_pairs = {(int(r["budget_minutes"]), int(r["seed"])) for r in tf_rows}
        for b in expected_budgets:
            for s in expected_seeds:
                if (int(b), int(s)) not in seen_pairs:
                    stale_benchmark = True
                    stale_reason = f"missing TF-1L-64d row for budget={b} seed={s}"
                    break
            if stale_benchmark:
                break

    rr_seed_rows = claim_table.get("rr_treelm", {}).get("per_seed", []) if isinstance(claim_table, dict) else []
    if stale_benchmark:
        report["primary"]["gate_2_crossover_exists"] = {"status": "invalid_stale", "reason": stale_reason}
        report["primary"]["gate_3_crossover_robust_2of3"] = {"status": "invalid_stale", "reason": stale_reason}
    elif rr_seed_rows and tf_rows:
        rr_by_seed = {int(r["seed"]): float(r["final_bpc"]) for r in rr_seed_rows if r.get("seed") is not None}
        by_budget: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in tf_rows:
            by_budget[int(r["budget_minutes"])].append(r)
        robustness_detail: Dict[str, Any] = {}
        crossover_any = False
        robust = False
        for budget, rows in by_budget.items():
            wins = 0
            total = 0
            for r in rows:
                seed = int(r["seed"])
                if seed not in rr_by_seed:
                    continue
                total += 1
                if rr_by_seed[seed] <= float(r["bpc"]):
                    wins += 1
            need = int(math.ceil((2.0 / 3.0) * max(1, total)))
            robustness_detail[str(budget)] = {"wins": wins, "total": total, "need": need}
            if wins > 0:
                crossover_any = True
            if wins >= need and total > 0:
                robust = True
        report["primary"]["gate_2_crossover_exists"] = {"status": "pass" if crossover_any else "fail"}
        report["primary"]["gate_3_crossover_robust_2of3"] = {"status": "pass" if robust else "fail", "detail": robustness_detail}
    elif rr_bpc is not None and tf_rows:
        crossover_any = any(float(rr_bpc) <= float(r["bpc"]) for r in tf_rows)
        by_budget: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in tf_rows:
            by_budget[int(r["budget_minutes"])].append(r)
        robust = False
        robustness_detail: Dict[str, Any] = {}
        for budget, rows in by_budget.items():
            wins = sum(1 for r in rows if float(rr_bpc) <= float(r["bpc"]))
            total = len(rows)
            need = int(math.ceil((2.0 / 3.0) * total))
            robustness_detail[str(budget)] = {"wins": wins, "total": total, "need": need}
            if wins >= need:
                robust = True
        report["primary"]["gate_2_crossover_exists"] = {
            "status": "pass" if crossover_any else "fail",
            "rr_bpc": float(rr_bpc),
        }
        report["primary"]["gate_3_crossover_robust_2of3"] = {
            "status": "pass" if robust else "fail",
            "detail": robustness_detail,
        }
    else:
        report["primary"]["gate_2_crossover_exists"] = {"status": "unknown"}
        report["primary"]["gate_3_crossover_robust_2of3"] = {"status": "unknown"}

    # Secondary 1: stage3 improves over stage1 on val.
    if metrics_val.get("stage3_nll") is not None and metrics_val.get("stage1_nll") is not None:
        s1 = float(metrics_val["stage1_nll"])
        s3 = float(metrics_val["stage3_nll"])
        report["secondary"]["stage3_better_than_stage1_val"] = {
            "status": "pass" if s3 < s1 else "fail",
            "stage1_nll": s1,
            "stage3_nll": s3,
        }
    else:
        report["secondary"]["stage3_better_than_stage1_val"] = {"status": "unknown"}

    # Secondary 2: retrieval_all better than retrieval_ppm_only.
    if isinstance(ablations, dict) and isinstance(ablations.get("results"), list):
        rmap = {r.get("ablation"): r for r in ablations["results"] if r.get("status") == "ok"}
        ra = rmap.get("retrieval_all")
        rp = rmap.get("retrieval_ppm_only")
        if ra and rp:
            report["secondary"]["retrieval_all_better_than_ppm_only"] = {
                "status": "pass" if float(ra["final_nll"]) < float(rp["final_nll"]) else "fail",
                "retrieval_all_nll": float(ra["final_nll"]),
                "ppm_only_nll": float(rp["final_nll"]),
            }
        else:
            report["secondary"]["retrieval_all_better_than_ppm_only"] = {"status": "unknown"}
    else:
        report["secondary"]["retrieval_all_better_than_ppm_only"] = {"status": "unknown"}

    # Secondary 3 requires extra datasets, mark unknown here.
    report["secondary"]["cross_dataset_gain_wikitext_enwik8"] = {"status": "unknown"}

    save_json(adir / "acceptance_report.json", report)
    print(json.dumps(report, indent=2))


def cmd_all(args: argparse.Namespace) -> None:
    dataset = args.dataset
    record_run_step(dataset, "all", args)
    thread_count = args.thread_count or default_thread_count()
    train_seeds = parse_csv_ints(args.train_seeds)
    frozen = {
        "dataset": dataset,
        "thread_count_frozen": int(thread_count),
        "train_seeds": train_seeds,
        "benchmark_budgets": parse_csv_ints(args.budgets),
        "benchmark_seeds": parse_csv_ints(args.benchmark_seeds),
        "fusion_mode": args.fusion_mode,
        "calibrator_mode": args.calibrator_mode,
        "ppm_alpha": args.ppm_alpha,
        "stage_beta": args.stage_beta,
        "simhash_gamma": args.simhash_gamma,
        "overlap_delta": args.overlap_delta,
        "calibrator_epsilon": args.calibrator_epsilon,
        "calibrator_clamp": args.calibrator_clamp,
        "aux_depth_threshold": args.aux_depth_threshold,
        "aux_entropy_threshold": args.aux_entropy_threshold,
        "created_at": now_iso(),
        "environment": build_env_metadata(thread_count=thread_count),
    }
    write_frozen_config(dataset, frozen)
    update_manifest(dataset, {"thread_count_frozen": int(thread_count), "train_seeds": train_seeds})

    cmd_prepare(argparse.Namespace(dataset=dataset))
    cmd_ppm_gate(
        argparse.Namespace(
            dataset=dataset,
            max_train_tokens=args.max_train_tokens,
            max_eval_tokens=args.max_eval_tokens,
        )
    )
    cmd_build_features(argparse.Namespace(dataset=dataset, max_examples=args.max_examples))
    cmd_build_retrieval(
        argparse.Namespace(
            dataset=dataset,
            max_examples=args.max_examples,
            max_train_tokens=args.max_train_tokens,
        )
    )
    rr_seed_rows: List[Dict[str, Any]] = []
    for seed in train_seeds:
        cmd_train_cascade(
            argparse.Namespace(
                dataset=dataset,
                seed=seed,
                thread_count=thread_count,
                max_examples=args.max_examples,
                iteration_scale=args.iteration_scale,
            )
        )
        cmd_train_calibrator(
            argparse.Namespace(
                dataset=dataset,
                seed=seed,
                thread_count=thread_count,
                max_examples=args.max_examples,
                candidate_pool=args.candidate_pool,
                calibrator_cap=args.calibrator_cap,
                tune_max_examples=args.tune_max_examples,
                fusion_mode=args.fusion_mode,
                calibrator_mode=args.calibrator_mode,
                ppm_alpha=args.ppm_alpha,
                stage_beta=args.stage_beta,
                simhash_gamma=args.simhash_gamma,
                overlap_delta=args.overlap_delta,
                calibrator_epsilon=args.calibrator_epsilon,
                calibrator_clamp=args.calibrator_clamp,
                aux_depth_threshold=args.aux_depth_threshold,
                aux_entropy_threshold=args.aux_entropy_threshold,
            )
        )
        val_metrics = evaluate_model(
            dataset=dataset,
            split="val",
            max_eval_examples=args.max_eval_examples,
            candidate_cap=args.candidate_pool,
            calibrator_cap=args.calibrator_cap,
            use_stage=args.use_stage,
        )
        test_metrics = evaluate_model(
            dataset=dataset,
            split="test",
            max_eval_examples=args.max_eval_examples,
            candidate_cap=args.candidate_pool,
            calibrator_cap=args.calibrator_cap,
            use_stage=args.use_stage,
        )
        rr_seed_rows.append({"seed": int(seed), **test_metrics})
        save_json(artifact_dir(dataset) / f"metrics_seed_{seed}_val.json", val_metrics)
        save_json(artifact_dir(dataset) / f"metrics_seed_{seed}_test.json", test_metrics)

    # Keep compatibility artifacts from last seed model.
    cmd_eval(
        argparse.Namespace(
            dataset=dataset,
            split="val",
            max_eval_examples=args.max_eval_examples,
            candidate_pool=args.candidate_pool,
            calibrator_cap=args.calibrator_cap,
            use_stage=args.use_stage,
        )
    )
    cmd_eval(
        argparse.Namespace(
            dataset=dataset,
            split="test",
            max_eval_examples=args.max_eval_examples,
            candidate_pool=args.candidate_pool,
            calibrator_cap=args.calibrator_cap,
            use_stage=args.use_stage,
        )
    )

    sample_seed = train_seeds[0] if train_seeds else args.seed
    for prompt in DEFAULT_PROMPTS:
        cmd_sample(
            argparse.Namespace(
                dataset=dataset,
                prompt=prompt,
                tokens=args.sample_tokens,
                temperature=0.9,
                top_p=0.95,
                seed=sample_seed,
                fusion_mode=args.fusion_mode,
                calibrator_mode=args.calibrator_mode,
                ppm_alpha=args.ppm_alpha,
                stage_beta=args.stage_beta,
                simhash_gamma=args.simhash_gamma,
                overlap_delta=args.overlap_delta,
                calibrator_epsilon=args.calibrator_epsilon,
                calibrator_clamp=args.calibrator_clamp,
                aux_depth_threshold=args.aux_depth_threshold,
                aux_entropy_threshold=args.aux_entropy_threshold,
                candidate_pool=args.candidate_pool,
                calibrator_cap=args.calibrator_cap,
                use_stage=args.use_stage,
            )
        )

    if args.with_ablations:
        cfg = str(CONFIG_DIR / "ablations.json") if (CONFIG_DIR / "ablations.json").exists() else None
        cmd_ablate(
            argparse.Namespace(
                dataset=dataset,
                split="test",
                config=cfg,
                max_eval_examples=args.max_eval_examples,
                retrain_required=args.retrain_required,
                seed=train_seeds[0] if train_seeds else args.seed,
                thread_count=args.thread_count,
                max_examples=args.max_examples,
                iteration_scale=args.iteration_scale,
                candidate_pool=args.candidate_pool,
                calibrator_cap=args.calibrator_cap,
                tune_max_examples=args.tune_max_examples,
                fusion_mode=args.fusion_mode,
                calibrator_mode=args.calibrator_mode,
                ppm_alpha=args.ppm_alpha,
                stage_beta=args.stage_beta,
                simhash_gamma=args.simhash_gamma,
                overlap_delta=args.overlap_delta,
                calibrator_epsilon=args.calibrator_epsilon,
                calibrator_clamp=args.calibrator_clamp,
                aux_depth_threshold=args.aux_depth_threshold,
                aux_entropy_threshold=args.aux_entropy_threshold,
            )
        )

    if args.with_benchmark:
        cmd_benchmark(
            argparse.Namespace(
                suite="crossover_cpu",
                thread_count=args.thread_count,
                max_eval_examples=args.max_eval_examples,
                budgets=args.budgets,
                seeds=args.benchmark_seeds,
            )
        )
        bpath = artifact_dir(dataset) / "benchmark_crossover_cpu.json"
        if bpath.exists():
            write_claim_table(dataset, load_json(bpath), rr_seed_rows=rr_seed_rows)
    else:
        # Even without transformer benchmark, persist seed summary table.
        write_claim_table(dataset, {"tiny_transformer": [], "rr_treelm": rr_seed_rows[-1] if rr_seed_rows else {}}, rr_seed_rows=rr_seed_rows)

    cmd_check_acceptance(
        argparse.Namespace(
            dataset=dataset,
            benchmark_path=(str(artifact_dir(dataset) / "benchmark_crossover_cpu.json") if args.with_benchmark else "__none__"),
            expected_budgets=args.budgets if args.with_benchmark else "",
            expected_seeds=args.benchmark_seeds if args.with_benchmark else "",
        )
    )


def write_claim_table(dataset: str, benchmark_payload: Mapping[str, Any], rr_seed_rows: Optional[List[Dict[str, Any]]] = None) -> None:
    adir = artifact_dir(dataset)
    tf_rows = list(benchmark_payload.get("tiny_transformer", []))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in tf_rows:
        if row.get("bpc") is None:
            continue
        key = f"{row.get('name')}|{row.get('budget_minutes')}"
        grouped[key].append(row)

    tf_agg = []
    for key, rows in grouped.items():
        name, budget = key.split("|", 1)
        nll = np.array([float(r["nll"]) for r in rows], dtype=np.float64)
        bpc = np.array([float(r["bpc"]) for r in rows], dtype=np.float64)
        tf_agg.append(
            {
                "name": name,
                "budget_minutes": int(budget),
                "num_seeds": int(len(rows)),
                "nll_mean": float(np.mean(nll)),
                "nll_std": float(np.std(nll)),
                "bpc_mean": float(np.mean(bpc)),
                "bpc_std": float(np.std(bpc)),
            }
        )

    rr = benchmark_payload.get("rr_treelm", {})
    rr_seeds = rr_seed_rows or []
    rr_summary = {
        "single_run": rr,
        "per_seed": rr_seeds,
    }
    if rr_seeds:
        rr_bpc = np.array([float(r["final_bpc"]) for r in rr_seeds], dtype=np.float64)
        rr_nll = np.array([float(r["final_nll"]) for r in rr_seeds], dtype=np.float64)
        rr_summary["aggregate"] = {
            "num_seeds": int(len(rr_seeds)),
            "final_nll_mean": float(np.mean(rr_nll)),
            "final_nll_std": float(np.std(rr_nll)),
            "final_bpc_mean": float(np.mean(rr_bpc)),
            "final_bpc_std": float(np.std(rr_bpc)),
        }

    payload = {
        "dataset": dataset,
        "created_at": now_iso(),
        "rr_treelm": rr_summary,
        "tiny_transformer": {"per_seed": tf_rows, "aggregate": sorted(tf_agg, key=lambda x: (x["name"], x["budget_minutes"]))},
    }
    save_json(adir / "claim_table.json", payload)


def cmd_benchmark(args: argparse.Namespace) -> None:
    record_run_step("tinyshakespeare", "benchmark", args)
    if args.suite != "crossover_cpu":
        raise ValueError("Only crossover_cpu suite is supported")

    dataset = "tinyshakespeare"
    adir = artifact_dir(dataset)
    vocab = load_vocab(dataset)
    splits = load_text_splits(dataset)

    train_ids = vocab.encode(splits["train"])
    val_ids = vocab.encode(splits["val"])
    test_ids = vocab.encode(splits["test"])

    thread_count = args.thread_count or default_thread_count()
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]

    results: Dict[str, Any] = {
        "suite": args.suite,
        "dataset": dataset,
        "thread_count": thread_count,
        "created_at": now_iso(),
        "budgets": budgets,
        "seeds": seeds,
        "rr_treelm": {},
        "tiny_transformer": [],
    }

    # RR-TreeLM metrics (expects model already trained)
    try:
        rr = evaluate_model(dataset, "test", max_eval_examples=args.max_eval_examples)
        results["rr_treelm"] = rr
    except Exception as e:
        results["rr_treelm"] = {"error": str(e)}

    update_manifest(
        dataset,
        {
            "benchmark_expectations": {
                "budgets": budgets,
                "seeds": seeds,
                "thread_count": thread_count,
                "updated_at": now_iso(),
            }
        },
    )
    tf_cfgs = [
        ("TF-1L-64d", 1, 64),
        ("TF-2L-96d", 2, 96),
    ]

    for name, nl, ne in tf_cfgs:
        for b in budgets:
            for seed in seeds:
                try:
                    metrics = run_tiny_transformer_budget(
                        train_ids=train_ids,
                        val_ids=val_ids,
                        test_ids=test_ids,
                        vocab_size=vocab.size,
                        block_size=256,
                        n_layer=nl,
                        n_embd=ne,
                        budget_minutes=b,
                        seed=seed,
                        thread_count=thread_count,
                    )
                    metrics["name"] = name
                    results["tiny_transformer"].append(metrics)
                except Exception as e:
                    results["tiny_transformer"].append(
                        {
                            "name": name,
                            "budget_minutes": b,
                        "seed": seed,
                        "error": str(e),
                    }
                )

    save_json(adir / "benchmark_crossover_cpu.json", results)
    write_claim_table(dataset, results)
    print(json.dumps(results, indent=2))


def cmd_benchmark_tf_shard(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset != "tinyshakespeare":
        raise ValueError("benchmark-tf-shard currently supports only tinyshakespeare")
    record_run_step(dataset, "benchmark-tf-shard", args)

    cfg_map = {
        "TF-1L-64d": (1, 64),
        "TF-2L-96d": (2, 96),
    }
    if args.model_name not in cfg_map:
        raise ValueError(f"Unknown model_name: {args.model_name}")
    n_layer, n_embd = cfg_map[args.model_name]

    thread_count = int(args.thread_count or default_thread_count())
    set_global_thread_env(thread_count)

    vocab = load_vocab(dataset)
    splits = load_text_splits(dataset)
    train_ids = vocab.encode(splits["train"])
    val_ids = vocab.encode(splits["val"])
    test_ids = vocab.encode(splits["test"])

    row = run_tiny_transformer_budget(
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        vocab_size=vocab.size,
        block_size=256,
        n_layer=int(n_layer),
        n_embd=int(n_embd),
        budget_minutes=int(args.budget_minutes),
        seed=int(args.seed),
        thread_count=thread_count,
    )
    row["name"] = args.model_name

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_dir = artifact_dir(dataset) / "tf_shards"
        ensure_dir(out_dir)
        safe_model = args.model_name.replace("-", "_")
        out_path = out_dir / f"{safe_model}_b{int(args.budget_minutes)}_s{int(args.seed)}.json"
    save_json(out_path, row)
    print(json.dumps({"output_json": str(out_path), "row": row}, indent=2))


def _load_json_rows_from_glob(glob_pattern: str) -> List[Tuple[Path, Dict[str, Any]]]:
    rows: List[Tuple[Path, Dict[str, Any]]] = []
    for pstr in sorted(glob.glob(glob_pattern, recursive=True)):
        p = Path(pstr)
        if not p.exists() or not p.is_file():
            continue
        try:
            payload = load_json(p)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append((p, payload))
    return rows


def _seed_from_metrics_path(path: Path, split: str) -> Optional[int]:
    m = re.search(rf"metrics_seed_(\d+)_{re.escape(split)}\.json$", str(path))
    if not m:
        return None
    return int(m.group(1))


def _mean_metric_rows(rows: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    if not rows:
        return {}
    out: Dict[str, Any] = {
        "dataset": rows[0].get("dataset", "tinyshakespeare"),
        "split": split,
        "count": int(rows[0].get("count", 0)),
        "created_at": now_iso(),
    }
    num_keys = [
        "stage1_nll",
        "stage2_nll",
        "stage3_nll",
        "ppm_nll",
        "final_nll",
        "final_bpc",
        "final_ppl",
        "top1_acc",
        "ngram_nll",
        "ngram_bpc",
        "ngram_ppl",
    ]
    for k in num_keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = float(np.mean(np.array(vals, dtype=np.float64)))
    if "thread_count" in rows[0]:
        vals = [int(r.get("thread_count", 0)) for r in rows if r.get("thread_count") is not None]
        if vals:
            out["thread_count"] = int(round(float(np.mean(np.array(vals, dtype=np.float64)))))
    return out


def cmd_benchmark_assemble(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset != "tinyshakespeare":
        raise ValueError("benchmark-assemble currently supports only tinyshakespeare")
    record_run_step(dataset, "benchmark-assemble", args)

    adir = artifact_dir(dataset)
    ensure_dir(adir)

    budgets = parse_csv_ints(args.budgets)
    seeds = parse_csv_ints(args.seeds)
    thread_count = int(args.thread_count or default_thread_count())

    rr_test_pairs = _load_json_rows_from_glob(args.rr_test_glob)
    rr_val_pairs = _load_json_rows_from_glob(args.rr_val_glob)
    tf_pairs = _load_json_rows_from_glob(args.tf_glob)

    rr_test_rows: List[Dict[str, Any]] = []
    for p, row in rr_test_pairs:
        r = dict(row)
        seed = r.get("seed")
        if seed is None:
            seed = _seed_from_metrics_path(p, split="test")
        if seed is None:
            continue
        r["seed"] = int(seed)
        rr_test_rows.append(r)
    rr_test_rows = sorted(rr_test_rows, key=lambda x: int(x["seed"]))

    rr_val_rows: List[Dict[str, Any]] = []
    for p, row in rr_val_pairs:
        r = dict(row)
        seed = r.get("seed")
        if seed is None:
            seed = _seed_from_metrics_path(p, split="val")
        if seed is None:
            continue
        r["seed"] = int(seed)
        rr_val_rows.append(r)
    rr_val_rows = sorted(rr_val_rows, key=lambda x: int(x["seed"]))

    tf_rows = [dict(row) for _, row in tf_pairs if row.get("name") and row.get("budget_minutes") is not None and row.get("seed") is not None]
    tf_rows = sorted(tf_rows, key=lambda x: (str(x.get("name")), int(x.get("budget_minutes", 0)), int(x.get("seed", 0))))

    rr_test_mean = _mean_metric_rows(rr_test_rows, split="test")
    rr_val_mean = _mean_metric_rows(rr_val_rows, split="val")
    if rr_test_mean:
        save_json(adir / "metrics_test.json", rr_test_mean)
        save_json(adir / "metrics.json", rr_test_mean)
    if rr_val_mean:
        save_json(adir / "metrics_val.json", rr_val_mean)

    benchmark_payload: Dict[str, Any] = {
        "suite": "crossover_cpu",
        "dataset": dataset,
        "thread_count": thread_count,
        "created_at": now_iso(),
        "budgets": budgets,
        "seeds": seeds,
        "rr_treelm": rr_test_mean,
        "tiny_transformer": tf_rows,
    }
    save_json(adir / "benchmark_crossover_cpu.json", benchmark_payload)
    write_claim_table(dataset, benchmark_payload, rr_seed_rows=rr_test_rows)
    update_manifest(
        dataset,
        {
            "benchmark_expectations": {
                "budgets": budgets,
                "seeds": seeds,
                "thread_count": thread_count,
                "updated_at": now_iso(),
            }
        },
    )

    cmd_check_acceptance(
        argparse.Namespace(
            dataset=dataset,
            benchmark_path=str(adir / "benchmark_crossover_cpu.json"),
            expected_budgets=",".join(str(x) for x in budgets),
            expected_seeds=",".join(str(x) for x in seeds),
        )
    )

    print(
        json.dumps(
            {
                "dataset": dataset,
                "rr_test_rows": len(rr_test_rows),
                "rr_val_rows": len(rr_val_rows),
                "tf_rows": len(tf_rows),
                "benchmark_path": str(adir / "benchmark_crossover_cpu.json"),
                "claim_table_path": str(adir / "claim_table.json"),
                "acceptance_path": str(adir / "acceptance_report.json"),
            },
            indent=2,
        )
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RR-TreeLM v4 (PPM-first)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_prepare.set_defaults(fn=cmd_prepare)

    p_ppm = sub.add_parser("ppm-gate")
    p_ppm.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_ppm.add_argument("--max-train-tokens", type=int, default=None)
    p_ppm.add_argument("--max-eval-tokens", type=int, default=None)
    p_ppm.set_defaults(fn=cmd_ppm_gate)

    p_feat = sub.add_parser("build-features")
    p_feat.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_feat.add_argument("--max-examples", type=int, default=None)
    p_feat.set_defaults(fn=cmd_build_features)

    p_cas = sub.add_parser("train-cascade")
    p_cas.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_cas.add_argument("--seed", type=int, default=1337)
    p_cas.add_argument("--thread-count", type=int, default=None)
    p_cas.add_argument("--max-examples", type=int, default=None)
    p_cas.add_argument("--iteration-scale", type=float, default=1.0)
    p_cas.set_defaults(fn=cmd_train_cascade)

    p_ret = sub.add_parser("build-retrieval")
    p_ret.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_ret.add_argument("--max-examples", type=int, default=None)
    p_ret.add_argument("--max-train-tokens", type=int, default=None)
    p_ret.set_defaults(fn=cmd_build_retrieval)

    p_cal = sub.add_parser("train-calibrator")
    p_cal.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_cal.add_argument("--seed", type=int, default=1337)
    p_cal.add_argument("--thread-count", type=int, default=None)
    p_cal.add_argument("--max-examples", type=int, default=None)
    p_cal.add_argument("--candidate-pool", type=int, default=1024)
    p_cal.add_argument("--calibrator-cap", type=int, default=192)
    p_cal.add_argument("--tune-max-examples", type=int, default=2000)
    p_cal.add_argument("--fusion-mode", choices=("ppm_primary", "balanced"), default="ppm_primary")
    p_cal.add_argument("--calibrator-mode", choices=("delta", "full"), default="delta")
    p_cal.add_argument("--ppm-alpha", type=float, default=1.0)
    p_cal.add_argument("--stage-beta", type=float, default=0.35)
    p_cal.add_argument("--simhash-gamma", type=float, default=0.08)
    p_cal.add_argument("--overlap-delta", type=float, default=0.08)
    p_cal.add_argument("--calibrator-epsilon", type=float, default=0.2)
    p_cal.add_argument("--calibrator-clamp", type=float, default=0.35)
    p_cal.add_argument("--aux-depth-threshold", type=float, default=6.0)
    p_cal.add_argument("--aux-entropy-threshold", type=float, default=3.0)
    p_cal.set_defaults(fn=cmd_train_calibrator)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_eval.add_argument("--split", required=True, choices=("val", "test"))
    p_eval.add_argument("--max-eval-examples", type=int, default=None)
    p_eval.add_argument("--candidate-pool", type=int, default=1024)
    p_eval.add_argument("--calibrator-cap", type=int, default=192)
    p_eval.add_argument("--use-stage", choices=("stage1", "stage2", "stage3"), default="stage3")
    p_eval.set_defaults(fn=cmd_eval)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_sample.add_argument("--prompt", default="ROMEO:")
    p_sample.add_argument("--tokens", type=int, default=400)
    p_sample.add_argument("--temperature", type=float, default=0.9)
    p_sample.add_argument("--top-p", type=float, default=0.95)
    p_sample.add_argument("--seed", type=int, default=1337)
    p_sample.add_argument("--candidate-pool", type=int, default=1024)
    p_sample.add_argument("--calibrator-cap", type=int, default=192)
    p_sample.add_argument("--use-stage", choices=("stage1", "stage2", "stage3"), default="stage3")
    p_sample.add_argument("--fusion-mode", choices=("ppm_primary", "balanced"), default=None)
    p_sample.add_argument("--calibrator-mode", choices=("delta", "full"), default=None)
    p_sample.add_argument("--ppm-alpha", type=float, default=None)
    p_sample.add_argument("--stage-beta", type=float, default=None)
    p_sample.add_argument("--simhash-gamma", type=float, default=None)
    p_sample.add_argument("--overlap-delta", type=float, default=None)
    p_sample.add_argument("--calibrator-epsilon", type=float, default=None)
    p_sample.add_argument("--calibrator-clamp", type=float, default=None)
    p_sample.add_argument("--aux-depth-threshold", type=float, default=None)
    p_sample.add_argument("--aux-entropy-threshold", type=float, default=None)
    p_sample.set_defaults(fn=cmd_sample)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--suite", required=True, choices=("crossover_cpu",))
    p_bench.add_argument("--thread-count", type=int, default=None)
    p_bench.add_argument("--max-eval-examples", type=int, default=5000)
    p_bench.add_argument("--budgets", type=str, default="15,60,240")
    p_bench.add_argument("--seeds", type=str, default="1337,2027,9001")
    p_bench.set_defaults(fn=cmd_benchmark)

    p_bench_tf = sub.add_parser("benchmark-tf-shard")
    p_bench_tf.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_bench_tf.add_argument("--model-name", required=True, choices=("TF-1L-64d", "TF-2L-96d"))
    p_bench_tf.add_argument("--budget-minutes", type=int, required=True)
    p_bench_tf.add_argument("--seed", type=int, required=True)
    p_bench_tf.add_argument("--thread-count", type=int, default=None)
    p_bench_tf.add_argument("--output-json", type=str, default=None)
    p_bench_tf.set_defaults(fn=cmd_benchmark_tf_shard)

    p_bench_asm = sub.add_parser("benchmark-assemble")
    p_bench_asm.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_bench_asm.add_argument("--rr-test-glob", required=True, type=str)
    p_bench_asm.add_argument("--rr-val-glob", required=True, type=str)
    p_bench_asm.add_argument("--tf-glob", required=True, type=str)
    p_bench_asm.add_argument("--budgets", type=str, default="15,60,240")
    p_bench_asm.add_argument("--seeds", type=str, default="1337,2027,9001")
    p_bench_asm.add_argument("--thread-count", type=int, default=None)
    p_bench_asm.set_defaults(fn=cmd_benchmark_assemble)

    p_ablate = sub.add_parser("ablate")
    p_ablate.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_ablate.add_argument("--split", default="test", choices=("val", "test"))
    p_ablate.add_argument("--config", default=str(CONFIG_DIR / "ablations.json"))
    p_ablate.add_argument("--max-eval-examples", type=int, default=None)
    p_ablate.add_argument("--retrain-required", action="store_true")
    p_ablate.add_argument("--seed", type=int, default=1337)
    p_ablate.add_argument("--thread-count", type=int, default=None)
    p_ablate.add_argument("--max-examples", type=int, default=None)
    p_ablate.add_argument("--iteration-scale", type=float, default=1.0)
    p_ablate.add_argument("--candidate-pool", type=int, default=1024)
    p_ablate.add_argument("--calibrator-cap", type=int, default=192)
    p_ablate.add_argument("--tune-max-examples", type=int, default=2000)
    p_ablate.add_argument("--fusion-mode", choices=("ppm_primary", "balanced"), default="ppm_primary")
    p_ablate.add_argument("--calibrator-mode", choices=("delta", "full"), default="delta")
    p_ablate.add_argument("--ppm-alpha", type=float, default=1.0)
    p_ablate.add_argument("--stage-beta", type=float, default=0.35)
    p_ablate.add_argument("--simhash-gamma", type=float, default=0.08)
    p_ablate.add_argument("--overlap-delta", type=float, default=0.08)
    p_ablate.add_argument("--calibrator-epsilon", type=float, default=0.2)
    p_ablate.add_argument("--calibrator-clamp", type=float, default=0.35)
    p_ablate.add_argument("--aux-depth-threshold", type=float, default=6.0)
    p_ablate.add_argument("--aux-entropy-threshold", type=float, default=3.0)
    p_ablate.set_defaults(fn=cmd_ablate)

    p_accept = sub.add_parser("check-acceptance")
    p_accept.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_accept.add_argument("--benchmark-path", default=None)
    p_accept.add_argument("--expected-budgets", type=str, default="")
    p_accept.add_argument("--expected-seeds", type=str, default="")
    p_accept.set_defaults(fn=cmd_check_acceptance)

    p_all = sub.add_parser("all")
    p_all.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p_all.add_argument("--seed", type=int, default=1337)
    p_all.add_argument("--train-seeds", type=str, default="1337,2027,9001")
    p_all.add_argument("--thread-count", type=int, default=None)
    p_all.add_argument("--max-examples", type=int, default=None)
    p_all.add_argument("--max-train-tokens", type=int, default=None)
    p_all.add_argument("--max-eval-tokens", type=int, default=None)
    p_all.add_argument("--max-eval-examples", type=int, default=None)
    p_all.add_argument("--iteration-scale", type=float, default=1.0)
    p_all.add_argument("--candidate-pool", type=int, default=1024)
    p_all.add_argument("--calibrator-cap", type=int, default=192)
    p_all.add_argument("--tune-max-examples", type=int, default=2000)
    p_all.add_argument("--use-stage", choices=("stage1", "stage2", "stage3"), default="stage3")
    p_all.add_argument("--fusion-mode", choices=("ppm_primary", "balanced"), default="ppm_primary")
    p_all.add_argument("--calibrator-mode", choices=("delta", "full"), default="delta")
    p_all.add_argument("--ppm-alpha", type=float, default=1.0)
    p_all.add_argument("--stage-beta", type=float, default=0.35)
    p_all.add_argument("--simhash-gamma", type=float, default=0.08)
    p_all.add_argument("--overlap-delta", type=float, default=0.08)
    p_all.add_argument("--calibrator-epsilon", type=float, default=0.2)
    p_all.add_argument("--calibrator-clamp", type=float, default=0.35)
    p_all.add_argument("--aux-depth-threshold", type=float, default=6.0)
    p_all.add_argument("--aux-entropy-threshold", type=float, default=3.0)
    p_all.add_argument("--sample-tokens", type=int, default=200)
    p_all.add_argument("--with-ablations", action="store_true")
    p_all.add_argument("--with-benchmark", action="store_true")
    p_all.add_argument("--retrain-required", action="store_true")
    p_all.add_argument("--budgets", type=str, default="15,60,240")
    p_all.add_argument("--benchmark-seeds", type=str, default="1337,2027,9001")
    p_all.set_defaults(fn=cmd_all)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        args.fn(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
