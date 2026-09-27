"""Independent acceptance: reread pinned sources and verify staged split artifacts."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from aggregate_scperturb_context import sha256
from build_context_splits import (
    assign,
    control_role,
    count_blocks,
    matrix_hash,
    read_metadata,
)
from scipy import sparse


def audit(input_dir, work, context):
    cfg = json.loads((input_dir/'release.json').read_text())['contexts'][context]
    out = work/'outputs'/context
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['config_sha256'] == sha256(input_dir/'release.json')
    for name,d in manifest['files'].items():
        assert sha256(out/name) == d['sha256'], name
        assert (out/name).stat().st_size == d['bytes']
    assert sha256(input_dir/cfg['reference']) == cfg['reference_sha256']
    assert sha256(input_dir/cfg['coverage']) == cfg['coverage_sha256']
    f = pd.read_parquet(out/'cell_assignments.parquet')
    assert f.locator.is_unique
    a = ad.read_h5ad(out/'construct_split_pseudobulk.h5ad')
    b = ad.read_h5ad(out/'batch_role_split_controls.h5ad')
    for obj,name in [(a,'construct_split_pseudobulk.h5ad'),(b,'batch_role_split_controls.h5ad')]:
        assert sparse.isspmatrix_csr(obj.X)
        assert obj.X.dtype.kind in 'iu' and (obj.X.data > 0).all()
        assert np.array_equal(np.asarray(obj.X.sum(1)).ravel(), obj.obs.total_umis)
        assert matrix_hash(obj.X, obj.obs, obj.var) == manifest['files'][name]['logical_sha256']
    assert a.var.equals(b.var)
    labels = sorted(set(f.construct))
    lc = {s:i for i,s in enumerate(labels)}
    direct = np.zeros((len(labels), a.n_vars),dtype=np.int64)
    # Independent sparse membership sums, using stored assignments only after recomputing them.
    reconstructed = np.zeros(a.shape,dtype=np.int64)
    reconstructed_controls = np.zeros(b.shape,dtype=np.int64)
    ar = {(str(row.construct),int(row.split)):i for i,row in enumerate(a.obs.itertuples())}
    br = {(str(row.batch),str(row.control_role),int(row.split)):i for i,row in enumerate(b.obs.itertuples())}
    source_count = retained_count = 0
    for source in cfg['sources']:
        path = work/'sources'/context/source['name']
        assert sha256(path) == source['sha256']
        full = read_metadata(path,source,cfg)
        selected = full[full.eligible & (full.is_control | full.construct.isin(cfg['construct_targets']))]
        sf = f[f.source_file.eq(source['name'])].sort_values('source_row').reset_index(drop=True)
        assert list(sf.source_row) == list(selected.source_row)
        for col in ['locator','construct','target','batch','source_cell_id']:
            assert list(sf[col]) == list(selected[col])
        check = assign(sf,context)
        assert list(check.split) == list(sf.split)
        for _, group in sf.groupby(['batch','construct'],observed=True):
            n = group.split.value_counts().reindex(range(4),fill_value=0)
            assert n.max()-n.min() <= 1
        assert all(role == (control_role(s) if ctrl else 'perturbation') for s,ctrl,role in zip(sf.construct,sf.is_control,sf.control_role,strict=True))
        retained_mask = sf.target.isin(cfg['responder_targets']) | (sf.is_control & sf.batch.isin(cfg['responder_batches']))
        assert np.array_equal(retained_mask, sf.retain_cell)
        rp = out/'responder_cells'/f'{Path(source["name"]).stem}.h5ad'
        cells = ad.read_h5ad(rp,backed='r') if retained_mask.any() else None
        if cells is not None:
            assert list(cells.obs.locator) == list(sf.loc[retained_mask,'locator'])
            assert cells.var.equals(a.var)
        ix = sf.set_index('source_row')
        cursor = 0
        for first,last,x in count_blocks(path,cfg['kind'],a.n_vars):
            rows = ix.loc[ix.index.to_series().between(first,last-1)]
            y = x[rows.index.to_numpy()-first]
            assert np.array_equal(np.asarray(y.sum(1)).ravel(),rows.total_umis.to_numpy())
            assert np.array_equal(y.getnnz(1),rows.n_genes_observed.to_numpy())
            # Independent unsplit aggregate includes new genes without an old reference.
            for label,positions in rows.groupby('construct',observed=True).indices.items():
                direct[lc[label]] += np.asarray(y[positions].sum(0)).ravel()
            codes = [ar[(s,int(r))] for s,r in zip(rows.construct,rows.split,strict=True)]
            np.add.at(reconstructed, codes, y.toarray())
            ctrl = rows.is_control.to_numpy()
            sub = rows[ctrl]
            codes = [br[(s,role,int(r))] for s,role,r in zip(sub.batch,sub.control_role,sub.split,strict=True)]
            np.add.at(reconstructed_controls, codes, y[ctrl].toarray())
            keep = rows.retain_cell.to_numpy(); n = int(keep.sum())
            if n:
                assert (cells.X[cursor:cursor+n] != y[keep]).nnz == 0
                cursor += n
        if cells is not None:
            assert cursor == cells.n_obs
            cells.file.close()
        source_count += len(sf); retained_count += cursor
        print(json.dumps({'audit_context':context,'source':source['name'],'source_rows_and_retained_counts_exact':True}),flush=True)
    assert np.array_equal(a.X.toarray(), reconstructed)
    assert np.array_equal(b.X.toarray(), reconstructed_controls)
    for label,i in lc.items():
        assert np.array_equal(np.asarray(a.X[a.obs.construct.eq(label).to_numpy()].sum(0)).ravel(),direct[i])
    ref = ad.read_h5ad(input_dir/cfg['reference'])
    assert a.var.equals(ref.var)
    matched = 0
    for i,row in enumerate(ref.obs.itertuples()):
        if row.guide_target in lc:
            assert np.array_equal(direct[lc[row.guide_target]],ref.X[i].toarray().ravel()), row.guide_target
            assert int(f.construct.eq(row.guide_target).sum()) == int(row.n_cells)
            matched += 1
    expected = pd.read_parquet(input_dir/cfg['coverage'])
    expected = expected[expected.target.str.lower().eq('non-targeting') | expected.construct.isin(cfg['construct_targets'])]
    keys = ['construct','target','batch']
    pd.testing.assert_series_equal(f.groupby(keys,observed=True).size().sort_index(),expected.groupby(keys,observed=True).n_cells.sum().sort_index(),check_names=False,check_dtype=False)
    exp = pd.read_parquet(out/'batch_split_exposures.parquet')
    keys = ['construct','target','batch','split','control_role']
    observed = f.groupby(keys,observed=True).agg(n_cells=('locator','size'),total_umis=('total_umis','sum'))
    recorded = exp.set_index(keys)[['n_cells','total_umis']]
    pd.testing.assert_frame_equal(recorded.loc[observed.index].sort_index(),observed.sort_index(),check_dtype=False)
    assert recorded.drop(observed.index).eq(0).all().all()
    for role in ['D_training','null_evaluation','effect_baseline']:
        ctrl = f[f.control_role.eq(role)]
        assert np.array_equal(np.asarray(a.X[a.obs.construct.isin(ctrl.construct).to_numpy()].sum(0)).ravel(),np.asarray(b.X[b.obs.control_role.eq(role).to_numpy()].sum(0)).ravel())
    assert source_count == len(f) and retained_count == int(f.retain_cell.sum())
    result = {'status':'passed','context':context,'manifest_sha256':sha256(out/'manifest.json'),'selected_cells':source_count,'retained_cells':retained_count,'reference_constructs_exact':matched,'independent_full_source_reaggregation':True,'responder_source_rows_exact':True,'file_sha256s_verified':True,'control_role_margins_exact':True,'missing_baseline_strata':manifest['missing_baseline_strata'],'promotion':'not performed'}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('work',type=Path);parser.add_argument('context')
    args = parser.parse_args();audit(args.input,args.work,args.context)
