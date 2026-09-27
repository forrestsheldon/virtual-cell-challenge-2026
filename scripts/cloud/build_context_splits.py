"""Build the approved six-context split release from pinned raw-count sources."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from aggregate_scperturb_context import add_grouped, blocks, column, sha256
from scipy import sparse

ROLES = ['D_training', 'null_evaluation', 'effect_baseline']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def control_role(label):
    bucket = int.from_bytes(hashlib.sha256(('control-role-v1|' + label).encode()).digest()[:8], 'big') % 4
    return ['D_training', 'null_evaluation', 'effect_baseline', 'effect_baseline'][bucket]


def assign(frame, context):
    """Hash-order round robin in each exact construct/batch, including sparse strata."""
    frame = frame.copy()
    frame['split'] = -1
    locators = frame.locator.tolist()
    for (batch, construct), positions in frame.groupby(['batch', 'construct'], sort=True).indices.items():
        key = ['cell-split-v1', 0, context, batch, construct]
        rotation = int.from_bytes(hashlib.sha256(canonical(['rotation', key]).encode()).digest()[:8], 'big') % 4
        ordered = sorted(positions, key=lambda i: (hashlib.sha256(canonical([key, locators[i]]).encode()).digest(), locators[i]))
        frame.loc[frame.index[ordered], 'split'] = (np.arange(len(ordered)) + rotation) % 4
    assert frame.split.between(0, 3).all()
    return frame


def read_metadata(path, source, cfg):
    if cfg['kind'] == 'scperturb':
        with h5py.File(path, 'r') as h:
            f = pd.DataFrame({k: column(h, 'obs/' + v) for k, v in [('target', 'gene'), ('construct', 'guide_id'), ('batch', 'batch')]})
            f['source_cell_id'] = column(h, 'obs/' + h['obs'].attrs['_index'])
            for k in ['ncounts', 'ngenes', 'core_scale_factor']:
                if k in h['obs']:
                    f[k] = column(h, 'obs/' + k)
            f['eligible'] = True
    else:
        f = pq.read_table(path, columns=['gene_target', 'guide_target', 'sample', 'cell_barcode', 'pass_guide_filter', 'total_counts', 'n_genes_by_counts']).to_pandas().rename(columns={'gene_target': 'target', 'guide_target': 'construct', 'sample': 'batch', 'cell_barcode': 'source_cell_id'})
        f['eligible'] = f.pass_guide_filter.astype(bool)
    for k in ['target', 'construct', 'batch', 'source_cell_id']:
        f[k] = f[k].astype(str)
    f['source_row'] = np.arange(len(f), dtype=np.int64)
    f['source_file'] = source['name']
    f['source_sha256'] = source['sha256']
    f['locator'] = [canonical([source['sha256'], source['name'], int(i)]) for i in f.source_row]
    f['is_control'] = f.target.str.lower().eq('non-targeting')
    return f


def count_blocks(path, kind, n_genes):
    if kind == 'scperturb':
        with h5py.File(path, 'r') as h:
            yield from blocks(h['X'], h['X'].shape[0], n_genes, 512)
    else:
        first = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=['gene_token_id', 'gene_expression'], use_threads=False):
            tokens, expression = batch.column(0), batch.column(1)
            indices = tokens.values.to_numpy().astype(np.int32)
            values = expression.values.to_numpy()
            offsets = tokens.offsets.to_numpy().astype(np.int64)
            assert np.array_equal(offsets, expression.offsets.to_numpy())
            assert np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.floor(values)).all()
            assert (indices >= 0).all() and (indices < n_genes).all()
            x = sparse.csr_matrix((values.astype(np.int64), indices, offsets), shape=(len(batch), n_genes))
            x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
            yield first, first + len(batch), x
            first += len(batch)


def select_metadata(full, cfg):
    f = full[full.eligible & (full.is_control | full.construct.isin(cfg['construct_targets']))].copy()
    f['context'] = cfg['context']
    f['exact_construct_id'] = f.construct.map(lambda s: canonical([cfg['namespace'], s]))
    f['control_role'] = [control_role(s) if ctrl else 'perturbation' for s, ctrl in zip(f.construct, f.is_control, strict=True)]
    f['retain_cell'] = f.target.isin(cfg['responder_targets']) | (f.is_control & f.batch.isin(cfg['responder_batches']))
    return assign(f, cfg['context'])


def matrix_hash(x, obs, var):
    """Stable logical hash includes row metadata, native gene order and integer CSR."""
    x = sparse.csr_matrix(x); x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    h = hashlib.sha256()
    for value in [list(x.shape), obs.reset_index().to_json(orient='split', double_precision=15), var.reset_index().to_json(orient='split', double_precision=15)]:
        h.update(canonical(value).encode()); h.update(b'\n')
    for a in [x.indptr, x.indices, x.data]:
        h.update(np.asarray(a, dtype='<i8').tobytes())
    return h.hexdigest()


def write_matrix(path, x, obs, var):
    x = sparse.csr_matrix(x); x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    obs = obs.copy(); obs.index = pd.Index([str(i) for i in range(len(obs))], name='row_id')
    a = ad.AnnData(x, obs=obs, var=var.copy())
    logical = matrix_hash(x, obs, var)
    a.uns['logical_sha256'] = logical
    a.write_h5ad(path, compression='gzip', compression_opts=1)
    return logical


def expected_coverage(input_dir, cfg):
    f = pd.read_parquet(input_dir / cfg['coverage'])
    return f[f.target.str.lower().eq('non-targeting') | f.construct.isin(cfg['construct_targets'])].groupby(['construct', 'target', 'batch'], observed=True).n_cells.sum().sort_index()


def build(input_dir, work, cfg):
    context = cfg['context']
    out = work / 'outputs' / context
    out.mkdir(parents=True, exist_ok=False)
    (out / 'responder_cells').mkdir()
    ref = ad.read_h5ad(input_dir / cfg['reference'], backed='r')
    var = ref.var.copy(); ref.file.close()
    labels = sorted(set(cfg['construct_targets']) | set(cfg['controls']))
    batches = sorted(cfg['batches'])
    lc = {s: i for i, s in enumerate(labels)}; bc = {s: i for i, s in enumerate(batches)}
    n_genes = len(var)
    split = np.zeros((len(labels)*4, n_genes), dtype=np.int64)
    controls = np.zeros((len(batches)*12, n_genes), dtype=np.int64)
    assignments, source_records, logicals = [], [], {}
    for source in cfg['sources']:
        path = work / 'sources' / context / source['name']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-all-errors', '--connect-timeout', '60', '--max-time', '3600', '--output', str(path), source['url']], check=True)
        assert path.stat().st_size == source['bytes'], str(path)
        assert sha256(path) == source['sha256'], str(path)
        if source.get('md5'):
            with path.open('rb') as f:
                assert hashlib.file_digest(f, 'md5').hexdigest() == source['md5']
        if cfg['kind'] == 'scperturb':
            with h5py.File(path, 'r') as h:
                assert list(column(h, 'var/ensembl_id')) == list(var.index)
                assert list(column(h, 'var/gene_name')) == list(var.gene_name)
        else:
            assert list(var.gene_token_id) == list(range(n_genes))
        full = read_metadata(path, source, cfg)
        f = select_metadata(full, cfg)
        assert f.locator.is_unique
        lookup = np.full(len(full), -1, dtype=np.int64)
        lookup[f.source_row] = np.arange(len(f))
        f = f.reset_index(drop=True)
        f['total_umis'] = np.int64(0)
        f['n_genes_observed'] = np.int64(0)
        rows = np.array([lc[s]*4 + int(r) for s, r in zip(f.construct, f.split, strict=True)])
        retained = []
        for first, last, x in count_blocks(path, cfg['kind'], n_genes):
            if cfg['kind'] == 'scperturb':
                assert np.array_equal(np.asarray(x.sum(1)).ravel(), full.ncounts.iloc[first:last].to_numpy())
                assert np.array_equal(x.getnnz(1), full.ngenes.iloc[first:last].to_numpy())
            ids = lookup[first:last]; keep = ids >= 0; ids = ids[keep]
            y = x[keep]
            f.loc[ids, 'total_umis'] = np.asarray(y.sum(1)).ravel()
            f.loc[ids, 'n_genes_observed'] = y.getnnz(1)
            add_grouped(split, y, rows[ids])
            ctrl = f.is_control.to_numpy()[ids]
            if ctrl.any():
                sub = f.iloc[ids[ctrl]]
                crow = np.array([bc[b]*12 + ROLES.index(role)*4 + int(r) for b, role, r in zip(sub.batch, sub.control_role, sub.split, strict=True)])
                add_grouped(controls, y[ctrl], crow)
            retained.append(y[f.retain_cell.to_numpy()[ids]])
        retain_obs = f[f.retain_cell].copy()
        if len(retain_obs):
            relative = f'responder_cells/{Path(source["name"]).stem}.h5ad'
            logicals[relative] = write_matrix(out / relative, sparse.vstack(retained, format='csr'), retain_obs, var)
        assignments.append(f)
        source_records.append({**source, 'input_cells': len(full), 'selected_cells': len(f), 'responder_cells': len(retain_obs), 'excluded_failed_guide_filter': int((~full.eligible).sum()), 'excluded_outside_panel': int((full.eligible & ~full.is_control & ~full.construct.isin(cfg['construct_targets'])).sum())})
        print(canonical({'context': context, 'source': source['name'], 'selected_cells': len(f), 'retained_cells': len(retain_obs), 'source_verified': True}), flush=True)
    f = pd.concat(assignments, ignore_index=True)
    assert f.locator.is_unique
    actual = f.groupby(['construct', 'target', 'batch'], observed=True).size().sort_index()
    pd.testing.assert_series_equal(actual, expected_coverage(input_dir, cfg), check_names=False, check_dtype=False)
    f.to_parquet(out / 'cell_assignments.parquet', index=False)
    groups = ['construct', 'target', 'batch', 'split', 'control_role']
    exp = f.groupby(groups, observed=True).agg(n_cells=('locator', 'size'), total_umis=('total_umis', 'sum')).reset_index()
    # Explicit zeros for all four splits of each occupied construct/batch stratum.
    base = f[groups[:-2]+['control_role']].drop_duplicates()
    base = base.merge(pd.DataFrame({'split': range(4)}), how='cross')
    exp = base.merge(exp, on=groups, how='left').fillna({'n_cells': 0, 'total_umis': 0})
    exp[['n_cells', 'total_umis']] = exp[['n_cells', 'total_umis']].astype('int64')
    exp.sort_values(groups).to_parquet(out / 'batch_split_exposures.parquet', index=False)
    obs = pd.DataFrame([(s, r) for s in labels for r in range(4)], columns=['construct', 'split'])
    target_map = f.set_index('construct').target.to_dict()
    obs['target'] = obs.construct.map(target_map)
    obs['exact_construct_id'] = obs.construct.map(lambda s: canonical([cfg['namespace'], s]))
    n = f.groupby(['construct', 'split'], observed=True).size()
    obs['n_cells'] = [int(n.get((s, r), 0)) for s, r in zip(obs.construct, obs.split, strict=True)]
    obs['total_umis'] = split.sum(1)
    logicals['construct_split_pseudobulk.h5ad'] = write_matrix(out / 'construct_split_pseudobulk.h5ad', split, obs, var)
    co = pd.DataFrame([(b, role, r) for b in batches for role in ROLES for r in range(4)], columns=['batch', 'control_role', 'split'])
    n = f[f.is_control].groupby(['batch', 'control_role', 'split'], observed=True).size()
    co['n_cells'] = [int(n.get(tuple(row), 0)) for row in co.itertuples(index=False, name=None)]
    co['total_umis'] = controls.sum(1)
    logicals['batch_role_split_controls.h5ad'] = write_matrix(out / 'batch_role_split_controls.h5ad', controls, co, var)
    support = exp.merge(co[['batch', 'control_role', 'split', 'n_cells']].query('control_role == "effect_baseline"').drop(columns='control_role').rename(columns={'n_cells': 'baseline_cells'}), on=['batch', 'split'], how='left')
    support['matched_baseline_available'] = support.baseline_cells.gt(0) | support.total_umis.eq(0)
    support.to_parquet(out / 'matched_control_support.parquet', index=False)
    files = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha256(p), 'logical_sha256': logicals.get(str(p.relative_to(out)))} for p in sorted(out.rglob('*')) if p.is_file()}
    record = {'context': context, 'config_sha256': sha256(input_dir/'release.json'), 'sources': source_records, 'files': files, 'selected_cells': len(f), 'retained_cells': int(f.retain_cell.sum()), 'missing_baseline_strata': int((~support.matched_baseline_available).sum()), 'status': 'built; independent audit required'}
    (out/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path); parser.add_argument('work', type=Path); parser.add_argument('context')
    args = parser.parse_args()
    pa.set_cpu_count(2)
    release = json.loads((args.input/'release.json').read_text())
    build(args.input, args.work, release['contexts'][args.context])
