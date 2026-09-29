"""H1 test of the 2026 fidelity zero point: score IDENTICAL cells of the H1 control mean.

The official 2026 baseline is the control-mean profile emitted as identical (fractional) cells;
the H1 harness baseline resamples real cells. If identical cells are what put the 2026
fidelity zero at a coin flip, this prediction should read raw fidelity near 0.5 on H1 (one
context), against 0.067 for the harness baseline and ~0 for resampled controls.
Every cell is the pooled mean count vector of all 38,176 H1 controls (fractional, like the
official arm). Only the harness's integer-count check is bypassed; scoring is otherwise the
harness's own `score_source`. Superseded for building the scale by the harness's
tools/build_control_mean_baseline.py (guide-balanced); kept as the pooled-mean test.
Run with the harness venv:
  vcc2026-h1-benchmark/.venv/bin/python scripts/evaluation/score_tiled_control_mean.py OUT_DIR
"""

import hashlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from vcc_h1_eval import scorer
from vcc_h1_eval.bounded import RowSource
from vcc_h1_eval.cli import _rescale, _score_args
from vcc_h1_eval.paths import BenchmarkPaths

OUT = Path(sys.argv[1])


class TiledX:
    def __init__(self, profile):
        self.row = sparse.csr_matrix(profile[None, :].astype(np.float64))

    def __getitem__(self, idx):
        return sparse.vstack([self.row] * len(np.atleast_1d(idx)), format="csr")


class Tiled:
    def __init__(self, profile, genes):
        self.X = TiledX(profile)
        self.var_names = pd.Index(genes)


paths = BenchmarkPaths.resolve(None)
args = _score_args(type("A", (), {"output": OUT, "prediction": None})(), paths)
controls_data = scorer.open_controls(args)
genes = controls_data.var_names.astype(str).tolist()
total = np.zeros(len(genes))
for start in range(0, controls_data.n_obs, 5000):
    total += np.asarray(controls_data.X[start:start + 5000].sum(axis=0)).ravel()
profile = total / controls_data.n_obs
print(f"control mean: {profile.sum():.0f} counts/cell, {np.count_nonzero(profile)} nonzero genes", flush=True)

targets = scorer.read_scoring_contract(args)
labels = np.repeat(targets, scorer.CELLS_PER_TARGET)
source = RowSource(Tiled(profile, genes), np.zeros(len(labels), dtype=np.int64), labels)
controls = RowSource(controls_data, np.arange(controls_data.n_obs), np.repeat(scorer.CONTROL, controls_data.n_obs))
identity = hashlib.sha256(f"tiled-control-mean:{profile.tobytes().hex()[:64]}".encode()).hexdigest()
try:
    scorer.score_source(args, source, controls, targets, {"path": "tiled control mean (in memory)"}, identity)
finally:
    controls_data.file.close()
_rescale(OUT, "vcc2026-val-1", paths)
