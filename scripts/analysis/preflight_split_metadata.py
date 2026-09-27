"""Read only source metadata through bounded HTTP ranges; never read count arrays."""
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived/context_transfer_preflight/metadata'


class Ranges(io.RawIOBase):
    def __init__(self, url, size, budget=8 * 1024**2):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.transferred = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.position)
        done = 0
        while done < n:
            k = self.position // 65536
            start, end = k * 65536, min((k + 1) * 65536, self.size) - 1
            if k not in self.cache:
                length = end - start + 1
                if self.transferred + length > self.budget:
                    raise RuntimeError('metadata range budget exhausted')
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(req, timeout=45) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise RuntimeError('server did not honor exact range; refused body')
                    value = r.read(length + 1)
                self.transferred += len(value)
                if len(value) != length:
                    raise RuntimeError('unexpected range length')
                self.cache[k] = value
            block = self.cache[k]
            offset = self.position - start
            take = min(n - done, len(block) - offset)
            buffer[done:done + take] = block[offset:offset + take]
            self.position += take
            done += take
        return done


def col(h, key):
    x = h[key]
    if isinstance(x, h5py.Group):
        codes = x['codes'][...]
        assert (codes >= 0).all()
        return np.asarray(x['categories'].asstr()[...])[codes]
    return x.asstr()[...] if x.dtype.kind in 'OS' else x[...]


def scperturb(context):
    m = json.loads((ROOT / f'data/derived/context_atlas/final/{context}/{context}_manifest.json').read_text())
    reader = Ranges(m['source_url'] + '?download=1', m['source_size'])
    record = {'context': context, 'source_url': reader.url, 'source_sha256': m['source_sha256']}
    try:
        with h5py.File(reader, 'r') as h:
            record.update(obs_columns=list(h['obs']), var_columns=list(h['var']), shape=list(h['X'].shape), dtype=str(h['X'].dtype), chunks=h['X'].chunks)
            frame = pd.DataFrame({k: col(h, 'obs/' + v) for k, v in [('construct', 'guide_id'), ('target', 'gene'), ('batch', 'batch')]})
            frame['n_cells'] = 1
            frame['nnz'] = col(h, 'obs/ngenes').astype(np.int64)
            frame['total_umis'] = col(h, 'obs/ncounts').astype(np.int64)
            if 'core_scale_factor' in h['obs']:
                factors = col(h, 'obs/core_scale_factor')
                record['upstream_core_scale_factor'] = {'min': float(factors.min()), 'max': float(factors.max()), 'finite': bool(np.isfinite(factors).all()), 'semantics': 'present; not yet validated as capture-exposure size factor'}
            frame.groupby(['construct', 'target', 'batch'], observed=True)[['n_cells','nnz','total_umis']].sum().reset_index().to_parquet(OUT / f'{context}_coverage.parquet', index=False)
            record['n_cells'] = len(frame)
            record['status'] = 'complete'
    except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as e:
        record.update(status='unavailable', error=str(e))
    record['metadata_bytes_read'] = reader.transferred
    (OUT / f'{context}_metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def xatlas(path):
    m = json.loads(path.read_text())
    dest = path.with_suffix('.coverage.parquet')
    info = path.with_suffix('.metadata.json')
    if info.exists() and json.loads(info.read_text()).get('status') == 'complete':
        return json.loads(info.read_text())
    reader = Ranges(m['source_url'], m['source_size'], 2 * 1024**2)
    record = {'source_path': m['source_path'], 'source_sha256': m['source_sha256']}
    try:
        f = pq.ParquetFile(reader)
        record['schema'] = str(f.schema_arrow)
        record['row_groups'] = f.num_row_groups
        record['expression_columns'] = [{ 'name': c.path_in_schema, 'values': c.num_values, 'compressed_bytes': c.total_compressed_size, 'uncompressed_bytes': c.total_uncompressed_size} for i in range(f.num_row_groups) for j in range(f.metadata.row_group(i).num_columns) if (c := f.metadata.row_group(i).column(j)).path_in_schema.startswith(('gene_token_id', 'gene_expression'))]
        t = f.read(columns=['gene_target', 'guide_target', 'sample', 'pass_guide_filter'], use_threads=False).to_pandas()
        record['n_cells'] = len(t)
        assert len(t) == m['n_input_cells']
        record['passed_cells'] = int(t.pass_guide_filter.sum())
        t = t[t.pass_guide_filter.astype(bool)].rename(columns={'guide_target':'construct', 'gene_target':'target', 'sample':'batch'})
        t.groupby(['construct', 'target', 'batch'], observed=True).size().rename('n_cells').reset_index().to_parquet(dest, index=False)
        record['status'] = 'complete'
    except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as e:
        record.update(status='unavailable', error=str(e))
    record['metadata_bytes_read'] = reader.transferred
    info.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'scperturb':
        with ThreadPoolExecutor(max_workers=4) as pool:
            for x in pool.map(scperturb, ['K562_essential', 'RPE1', 'HepG2', 'Jurkat']):
                print(json.dumps(x), flush=True)
    else:
        paths = [p for c in ['HCT116', 'HEK293T'] for p in (OUT / c).glob('*_Batch*.json') if '.metadata.' not in p.name]
        if len(sys.argv) > 2:
            paths = paths[:int(sys.argv[2])]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(xatlas, p) for p in paths]
            for i, f in enumerate(as_completed(futures)):
                x = f.result()
                print(json.dumps({'i':i+1, **{k:v for k,v in x.items() if k != 'schema' and k != 'expression_columns'}}), flush=True)
