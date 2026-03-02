# RR-TreeLM v4 (PPM-First)

Execution-safe implementation of a retrieval-heavy, cascade-based, tree-only autoregressive character LM with a PPM-primary fusion path.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Pipeline

```bash
python3 treegpt.py prepare --dataset tinyshakespeare
python3 treegpt.py ppm-gate --dataset tinyshakespeare
python3 treegpt.py build-features --dataset tinyshakespeare
python3 treegpt.py train-cascade --dataset tinyshakespeare --seed 1337
python3 treegpt.py build-retrieval --dataset tinyshakespeare
python3 treegpt.py train-calibrator --dataset tinyshakespeare --seed 1337 --fusion-mode ppm_primary --calibrator-mode delta
python3 treegpt.py eval --dataset tinyshakespeare --split test --candidate-pool 1024 --calibrator-cap 192
python3 treegpt.py sample --dataset tinyshakespeare --prompt "ROMEO:" --tokens 400 --temperature 0.9 --top-p 0.95
python3 treegpt.py benchmark --suite crossover_cpu
python3 treegpt.py benchmark-tf-shard --dataset tinyshakespeare --model-name TF-1L-64d --budget-minutes 15 --seed 1337
python3 treegpt.py benchmark-assemble --dataset tinyshakespeare --rr-test-glob "ci_in/rr/**/metrics_seed_*_test.json" --rr-val-glob "ci_in/rr/**/metrics_seed_*_val.json" --tf-glob "ci_in/tf/**/*.json" --budgets 15,60,240 --seeds 1337,2027,9001
python3 treegpt.py ablate --dataset tinyshakespeare --split test --retrain-required
python3 treegpt.py check-acceptance --dataset tinyshakespeare --expected-budgets 15,60,240 --expected-seeds 1337,2027,9001
```

## Notes

- OOF folds are fixed at `K=3` in cascade training.
- For smoke tests, use `--max-examples`, `--max-train-tokens`, and `--iteration-scale` (for example `--iteration-scale 0.05`).
- For faster calibrator smoke runs, lower `--candidate-pool`, `--calibrator-cap`, and `--tune-max-examples`.
- `ppm-gate` writes `artifacts/tinyshakespeare/ppm_gate.json` and enforces the `BPC <= 1.52` decision rule in `train-cascade`.
- Collision policy is applied during `build-features`: hashed namespace dimensions are doubled if estimated collision rate exceeds `20%`.
- Candidate pruning is fixed at top `192` calibrator evaluations/token after cheap pre-score.
- v4 fusion controls:
  - `--fusion-mode {ppm_primary,balanced}`
  - `--calibrator-mode {delta,full}`
  - `--ppm-alpha --stage-beta --simhash-gamma --overlap-delta --calibrator-epsilon`
- Reproducibility artifacts:
  - `artifacts/<dataset>/run_manifest.json`
  - `artifacts/<dataset>/frozen_config.json`
  - `artifacts/<dataset>/claim_table.json`
- GitHub-hosted claim workflow:
  - `.github/workflows/claim-crossover.yml`
  - Shards transformer runs by `(model, budget, seed)` and RR runs by `seed`, then assembles:
    - `benchmark_crossover_cpu.json`
    - `claim_table.json`
    - `acceptance_report.json`
  - Designed for public-repo GitHub Actions execution (no local machine runtime required).
- `all` orchestrates end-to-end execution. Example smoke run:
  - `python3 treegpt.py all --dataset tinyshakespeare --train-seeds 1337 --max-examples 200 --max-train-tokens 20000 --max-eval-tokens 2000 --max-eval-examples 200 --iteration-scale 0.02 --sample-tokens 80 --with-ablations`
