"""Freeze the approved panel, metadata coverage and immutable references for the cloud build."""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT/'data/derived/context_transfer_preflight/metadata'
OUT = ROOT/'data/derived/context_transfer_splits/split-v1-20260924/input'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    r = json.loads((ROOT/'reports/context-transfer-blog/split-artifact-preflight.json').read_text())
    old = json.loads((ROOT/'reports/context-transfer-blog/archive/split-artifact-preflight-v1/split-artifact-preflight.json').read_text())
    frames = {c: pd.concat([pd.read_parquet(p) for p in sorted((META/c).glob('*.coverage.parquet'))], ignore_index=True) if c in ['HCT116','HEK293T'] else pd.read_parquet(META/f'{c}_coverage.parquet') for c in r['contexts']}
    for f in frames.values():
        f['batch'] = f.batch.astype(str)
    top = set(pd.read_csv(ROOT/'reports/context-transfer-blog/split-artifact-xatlas-top200.csv').target)
    panel = set(r['panel_targets']) | top
    release = {'release': 'split-v1-20260924', 'approval': 'Forrest requested launch on 2026-09-24 following responder retention update', 'project': 'bold-bastion-509200-f9', 'control_roles': 'control-role-v1 SHA256 modulo4: D_training/null_evaluation/effect_baseline/effect_baseline; same labels across contexts', 'missing_control_policy': 'Flag unavailable; no pooled substitution or cell filtering', 'batch_folds': 'not included', 'contexts': {}}
    for pair in r['pairs']:
        left,right = pair.split('__')
        shared = set(frames[left].construct) & set(frames[right].construct)
        for c in [left,right]:
            xatlas = c in ['HCT116','HEK293T']
            f = frames[c]
            controls = f.target.str.lower().eq('non-targeting')
            selected = f[~controls & f.construct.isin(shared)]
            if xatlas:
                selected = selected[selected.target.isin(panel)]
            mapping = selected[['construct','target']].drop_duplicates().set_index('construct').target.to_dict()
            assert len(mapping) == r['storage_profiles']['union415_plus_bottleneck_top200']['contexts'][c]['constructs']
            resp = old['pairs'][pair]['options']['union415' if xatlas else 'all_shared'][c]['multiconstruct_targets']
            rb = sorted(selected[selected.target.isin(resp)].batch.unique())
            retain = f[(f.construct.isin(mapping) & f.target.isin(resp)) | (controls & f.batch.isin(rb))]
            assert int(retain.n_cells.sum()) == sum(r['responder_retention']['contexts'][c][k] for k in ['responder_perturbed_cells','responder_all_controls'])
            family = 'xatlas_orion' if xatlas else 'context_atlas'
            refdir = ROOT/f'data/derived/{family}/final/{c}'
            refname = f'{c}_guide_pseudobulk.h5ad'
            refmanifest = json.loads((refdir/f'{c}_manifest.json').read_text())
            assert sha(refdir/refname) == refmanifest['outputs'][refname]['sha256']
            shutil.copyfile(refdir/refname, OUT/refname)
            coverage = f'{c}_coverage.parquet'; f.to_parquet(OUT/coverage,index=False)
            s = r['contexts'][c]['source']
            sources = [{'name':Path(m['source_path']).name,'url':m['source_url'],'bytes':m['source_size'],'sha256':m['source_sha256']} for m in s['shards']] if xatlas else [{'name':s['source_file'],'url':s.get('download_url',s['source_url']),'bytes':s['source_size'],'sha256':s['source_sha256'],'md5':s['source_md5']}]
            release['contexts'][c] = {'context':c,'namespace':pair,'kind':'xatlas' if xatlas else 'scperturb','construct_targets':mapping,'controls':sorted(f.loc[controls,'construct'].unique()),'batches':sorted(f.batch.unique()),'responder_targets':resp,'responder_batches':rb,'sources':sources,'coverage':coverage,'reference':refname,'reference_sha256':sha(OUT/refname),'coverage_sha256':sha(OUT/coverage)}
    (OUT/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    print(json.dumps({'config_sha256':sha(OUT/'release.json'),'contexts':{c:{'constructs':len(d['construct_targets']),'responder_targets':len(d['responder_targets']),'sources':len(d['sources'])} for c,d in release['contexts'].items()}},indent=2))


if __name__ == '__main__':
    main()
