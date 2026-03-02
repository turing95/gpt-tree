from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treegpt import (
    FusionWeights,
    _mean_metric_rows,
    _seed_from_metrics_path,
    compute_aux_gate,
    FeatureExtractor,
    FeatureSpec,
    PPMDModel,
    Vocabulary,
    combine_distribution,
    oof_split_indices,
    safe_softmax,
    select_candidates,
)


def test_hash_determinism_and_feature_stability():
    vocab = Vocabulary()
    spec = FeatureSpec()
    extractor = FeatureExtractor(spec, vocab)

    ctx = np.full(spec.ctx_len, vocab.bos_id, dtype=np.int16)
    txt = vocab.encode("ROMEO:\n")
    ctx[-len(txt) :] = txt

    idx1, vals1, _ = extractor.extract_indices_values(ctx)
    idx2, vals2, _ = extractor.extract_indices_values(ctx)

    assert idx1 == idx2
    assert np.allclose(vals1, vals2)


def test_feature_dim_bounds():
    vocab = Vocabulary()
    spec = FeatureSpec()
    extractor = FeatureExtractor(spec, vocab)
    ctx = np.full(spec.ctx_len, vocab.bos_id, dtype=np.int16)
    idx, _, _ = extractor.extract_indices_values(ctx)
    assert min(idx) >= 0
    assert max(idx) < spec.total_dim


def test_oof_split_integrity_no_leakage():
    n = 300
    splits = oof_split_indices(n=n, k_folds=3, seed=1337)
    seen_val = set()
    all_idx = set(range(n))

    for tr, va in splits:
        tr_set = set(tr.tolist())
        va_set = set(va.tolist())
        assert tr_set.isdisjoint(va_set)
        seen_val.update(va_set)

    assert seen_val == all_idx


def test_ppm_escape_depth_and_prob_normalization():
    vocab = Vocabulary()
    ids = vocab.encode("abcabcabcabc")
    ppm = PPMDModel(vocab_size=vocab.size, max_order=4)
    ppm.fit(ids)
    ctx = vocab.encode("abcab")

    dist, conf = ppm.predict_distribution(ctx)
    assert np.isclose(dist.sum(), 1.0, atol=1e-6)
    assert float(conf["max_match_depth"]) >= 1.0


def test_safe_softmax_finite_and_normalized():
    x = np.array([1000.0, 1001.0, -9999.0], dtype=np.float32)
    p = safe_softmax(x)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0, atol=1e-6)


def test_select_candidates_respects_cap():
    vocab = Vocabulary()
    v = vocab.size
    rng = np.random.default_rng(1337)
    base = rng.random(v).astype(np.float32)
    base /= base.sum()
    r = {
        "dist": {
            "ppm": np.sort(rng.random(v).astype(np.float32))[::-1],
            "simhash": np.sort(rng.random(v).astype(np.float32))[::-1],
            "overlap": np.sort(rng.random(v).astype(np.float32))[::-1],
        }
    }
    for k in r["dist"]:
        r["dist"][k] /= r["dist"][k].sum()
    cand = select_candidates(base, r, candidate_cap=64, calibrator_cap=32)
    assert len(cand) <= 32


def test_combine_distribution_keeps_tail_mass():
    class DummyCal:
        def predict(self, X):
            return np.zeros((X.shape[0],), dtype=np.float32)

        def predict_proba(self, X):
            n = X.shape[0]
            out = np.zeros((n, 2), dtype=np.float32)
            out[:, 1] = 0.5
            out[:, 0] = 0.5
            return out

    vocab = Vocabulary()
    v = vocab.size
    base = np.full(v, 1.0 / v, dtype=np.float32)
    r = {
        "dist": {
            "ppm": np.full(v, 1.0 / v, dtype=np.float32),
            "simhash": np.full(v, 1.0 / v, dtype=np.float32),
            "overlap": np.full(v, 1.0 / v, dtype=np.float32),
        },
        "conf": {"ppm": {}, "simhash": {}, "overlap": {}},
    }
    stage_probs = {"stage1": base.copy(), "stage2": base.copy(), "stage3": base.copy()}
    ctx = np.full(256, vocab.bos_id, dtype=np.int16)
    p = combine_distribution(
        stage3=base,
        retr=r,
        calibrator_model=DummyCal(),
        context=ctx,
        stage_probs=stage_probs,
        fusion=FusionWeights(),
        candidate_cap=16,
        calibrator_cap=8,
    )
    assert np.isclose(p.sum(), 1.0, atol=1e-6)
    # Tail should keep non-zero mass by construction.
    top = np.argsort(-p)[:8]
    tail_mass = float(np.sum(np.delete(p, top)))
    assert tail_mass > 0.0


def test_delta_calibrator_is_clamped():
    class WildDeltaCal:
        def predict(self, X):
            # Simulate extreme delta predictions; combine_distribution should clamp this.
            return np.full((X.shape[0],), 1000.0, dtype=np.float32)

    vocab = Vocabulary()
    v = vocab.size
    base = np.full(v, 1.0 / v, dtype=np.float32)
    r = {
        "dist": {
            "ppm": np.full(v, 1.0 / v, dtype=np.float32),
            "simhash": np.full(v, 1.0 / v, dtype=np.float32),
            "overlap": np.full(v, 1.0 / v, dtype=np.float32),
        },
        "conf": {"ppm": {}, "simhash": {}, "overlap": {}},
    }
    stage_probs = {"stage1": base.copy(), "stage2": base.copy(), "stage3": base.copy()}
    ctx = np.full(256, vocab.bos_id, dtype=np.int16)
    p = combine_distribution(
        stage3=base,
        retr=r,
        calibrator_model=WildDeltaCal(),
        context=ctx,
        stage_probs=stage_probs,
        fusion=FusionWeights(calibrator_mode="delta", calibrator_epsilon=0.2, calibrator_clamp=0.35),
        candidate_cap=16,
        calibrator_cap=8,
    )
    assert np.isclose(p.sum(), 1.0, atol=1e-6)
    assert float(np.max(p)) < 0.9


def test_confidence_gate_reduces_aux_weight():
    strong = compute_aux_gate({"ppm": {"max_match_depth": 8.0, "entropy": 2.5}}, depth_threshold=6.0, entropy_threshold=3.0)
    weak = compute_aux_gate({"ppm": {"max_match_depth": 1.0, "entropy": 5.0}}, depth_threshold=6.0, entropy_threshold=3.0)
    assert strong < weak


def test_seed_from_metrics_path_extracts_seed():
    p = Path("/tmp/foo/metrics_seed_2027_test.json")
    assert _seed_from_metrics_path(p, "test") == 2027
    assert _seed_from_metrics_path(p, "val") is None


def test_mean_metric_rows_computes_means():
    rows = [
        {"dataset": "tinyshakespeare", "count": 10, "final_nll": 2.0, "stage3_nll": 3.0, "ngram_nll": 1.0, "thread_count": 2},
        {"dataset": "tinyshakespeare", "count": 10, "final_nll": 4.0, "stage3_nll": 5.0, "ngram_nll": 3.0, "thread_count": 2},
    ]
    out = _mean_metric_rows(rows, split="test")
    assert out["dataset"] == "tinyshakespeare"
    assert out["split"] == "test"
    assert np.isclose(out["final_nll"], 3.0)
    assert np.isclose(out["stage3_nll"], 4.0)
    assert np.isclose(out["ngram_nll"], 2.0)
