import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).parents[1]


def load(name):
    path = ROOT / "scripts/cloud" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(monkeypatch, module, arguments):
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, arguments)])
    module.main()


def test_kolf_pooling_excludes_unrequested_target(tmp_path, monkeypatch):
    module = load("aggregate_kolf_h5ad.py")
    counts = np.array([[1, 2, 0], [0, 1, 3], [2, 0, 1], [9, 9, 9]])
    data = ad.AnnData(
        X=sparse.csc_matrix(counts),
        obs=pd.DataFrame(
            {
                "gene_target": pd.Categorical(["A", "A", "NTC", "B"]),
                "gRNA": pd.Categorical(["A_1", "A_2", "NTC_1", "B_1"]),
                "batch": pd.Categorical(["one", "two", "one", "one"]),
                "channel": pd.Categorical(["c1", "c2", "c1", "c1"]),
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame({"gene_ids": ["e1", "e2", "e3"]}, index=["g1", "g2", "g3"]),
    )
    data.layers["counts"] = sparse.csc_matrix(counts)
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    targets = tmp_path / "targets.csv"
    targets.write_text("target_gene\nA\n")
    output = tmp_path / "out"

    run_main(
        monkeypatch,
        module,
        [
            source,
            "--targets",
            targets,
            "--source-md5",
            "test",
            "--source-sha256",
            "test",
            "--output",
            output,
            "--gene-block",
            "2",
        ],
    )

    result = ad.read_h5ad(output / "KOLF2.1J_target_pseudobulk.h5ad")
    np.testing.assert_array_equal(result["A"].X.toarray(), [[1, 3, 3]])
    np.testing.assert_array_equal(result["NTC"].X.toarray(), [[2, 0, 1]])
    assert result.obs["n_cells"].to_dict() == {"A": 2, "NTC": 1}
    run_main(monkeypatch, load("audit_context_pseudobulks.py"), [output])


def test_cd4_pooling_and_contiguous_runs(tmp_path, monkeypatch):
    module = load("aggregate_cd4_pseudobulk.py")
    assert module.contiguous_runs(np.array([1, 2, 5, 9, 10])) == [
        (1, 3),
        (5, 6),
        (9, 11),
    ]
    counts = np.array([[1, 2, 0], [0, 1, 3], [2, 0, 1], [9, 9, 9]])
    data = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(
            {
                "perturbed_gene_name": pd.Categorical(["A", "A", "NTC", "B"]),
                "guide_id": pd.Categorical(["A-1", "A-2", "NTC-1", "B-1"]),
                "guide_type": pd.Categorical(
                    ["targeting", "targeting", "non-targeting", "targeting"]
                ),
                "donor_id": pd.Categorical(["d1", "d2", "d1", "d1"]),
                "culture_condition": pd.Categorical(["rest", "stim", "rest", "rest"]),
                "n_cells": [2.0, 3.0, 4.0, 5.0],
                "total_counts": counts.sum(1).astype(float),
                "keep_for_DE": [True, True, True, True],
                "keep_effective_guides": [True, False, True, True],
            },
            index=["p1", "p2", "p3", "p4"],
        ),
        var=pd.DataFrame(
            {
                "gene_ids": ["e1", "e2", "e3"],
                "gene_name": ["g1", "g2", "g3"],
            },
            index=["e1", "e2", "e3"],
        ),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    targets = tmp_path / "targets.csv"
    targets.write_text("target_gene\nA\n")
    output = tmp_path / "out"

    run_main(
        monkeypatch,
        module,
        [
            source,
            "--targets",
            targets,
            "--source-sha256",
            "test",
            "--output",
            output,
        ],
    )

    result = ad.read_h5ad(output / "CD4T_target_pseudobulk.h5ad")
    np.testing.assert_array_equal(result["A"].X.toarray(), [[1, 3, 3]])
    np.testing.assert_array_equal(result["NTC"].X.toarray(), [[2, 0, 1]])
    assert result.obs["n_cells"].to_dict() == {"A": 5, "NTC": 4}
    run_main(monkeypatch, load("audit_context_pseudobulks.py"), [output])


def test_cd4_rejects_mismatched_control_annotation(tmp_path, monkeypatch):
    module = load("aggregate_cd4_pseudobulk.py")
    data = ad.AnnData(
        X=sparse.csr_matrix([[1]]),
        obs=pd.DataFrame(
            {
                "perturbed_gene_name": pd.Categorical(["NTC"]),
                "guide_id": pd.Categorical(["NTC-1"]),
                "guide_type": pd.Categorical(["targeting"]),
                "donor_id": pd.Categorical(["d1"]),
                "culture_condition": pd.Categorical(["rest"]),
                "n_cells": [1.0],
                "total_counts": [1.0],
                "keep_for_DE": [True],
                "keep_effective_guides": [True],
            },
            index=["p1"],
        ),
        var=pd.DataFrame({"gene_ids": ["e1"], "gene_name": ["g1"]}, index=["e1"]),
    )
    source = tmp_path / "bad.h5ad"
    data.write_h5ad(source)
    targets = tmp_path / "targets.csv"
    targets.write_text("target_gene\nA\n")

    with pytest.raises(ValueError, match="NTC labels"):
        run_main(
            monkeypatch,
            module,
            [
                source,
                "--targets",
                targets,
                "--source-sha256",
                "test",
                "--output",
                tmp_path / "out",
            ],
        )


@pytest.mark.parametrize("sparse_x", [False, True])
def test_scperturb_full_panel_dense_and_csr(tmp_path, monkeypatch, sparse_x):
    module = load("aggregate_scperturb_context.py")
    counts = np.array([[1, 2, 0], [0, 1, 3], [2, 0, 1], [9, 9, 9]])
    matrix = sparse.csr_matrix(counts) if sparse_x else counts.astype(np.float32)
    data = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                "gene": pd.Categorical(["A", "A", "non-targeting", "B"]),
                "guide_id": pd.Categorical(["A-1", "A-2", "NT-1", "B-1"]),
                "batch": [1, 2, 1, 1],
                "ncounts": counts.sum(1).astype(float),
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(
            {
                "ensembl_id": ["e1", "e2", "e3"],
                "gene_name": ["g1", "g2", "g3"],
            },
            index=["g1", "g2", "g3"],
        ),
    )
    source = tmp_path / "source.h5ad"
    data.write_h5ad(source)
    output = tmp_path / "out"
    run_main(
        monkeypatch,
        module,
        [
            source,
            "--context",
            "TEST",
            "--dataset",
            "test",
            "--source-url",
            "https://example.test/source",
            "--source-md5",
            "test",
            "--source-sha256",
            "test",
            "--output",
            output,
            "--block-size",
            "2",
        ],
    )
    result = ad.read_h5ad(output / "TEST_target_pseudobulk.h5ad")
    np.testing.assert_array_equal(result["A"].X.toarray(), [[1, 3, 3]])
    np.testing.assert_array_equal(
        result["non-targeting"].X.toarray(), [[2, 0, 1]]
    )
    np.testing.assert_array_equal(result["B"].X.toarray(), [[9, 9, 9]])
    run_main(monkeypatch, load("audit_context_pseudobulks.py"), [output])
