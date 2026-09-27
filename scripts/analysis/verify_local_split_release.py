"""Verify downloaded split release against cloud hash chains and compact references."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/derived/context_transfer_splits/split-v1-20260924'
REPORT = ROOT/'reports/context-transfer-blog/split-v1-20260924/completion'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(core_only=False):
    status=json.loads((REPORT/'status.json').read_text())
    assert status['state']=='completed_staging_validated'
    config=json.loads((DATA/'input/release.json').read_text())
    result={'status':'passed','local_scope':'compact core' if core_only else 'all derived artifacts','contexts':{},'total_artifact_bytes':0,'total_local_verified_bytes':0,'total_retained_cells':0,'cloud_prefix':'gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/','local_root':str(DATA)}
    for c,entry in status['artifacts'].items():
        dest=DATA/c
        assert sha(dest/'transport-audit.json')==entry['sha256']
        transport=json.loads((dest/'transport-audit.json').read_text())
        assert transport['status']=='passed'
        files={}
        for item in transport['files']:
            name=item['uri'].split('/'+c+'/',1)[1]
            p=dest/name
            if not (core_only and name.startswith('responder_cells/')):
                assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
            files[name]={'bytes':item['bytes'],'sha256':item['sha256'],'verified_locally':not (core_only and name.startswith('responder_cells/'))}
        manifest=json.loads((dest/'manifest.json').read_text())
        acceptance=json.loads((dest/'acceptance.json').read_text())
        assert acceptance['status']=='passed'
        assert acceptance['manifest_sha256']==sha(dest/'manifest.json')
        assert manifest['config_sha256']==sha(DATA/'input/release.json')
        for name,d in manifest['files'].items():
            assert d['sha256']==files[name]['sha256']
        f=pd.read_parquet(dest/'cell_assignments.parquet')
        assert f.locator.is_unique
        a=ad.read_h5ad(dest/'construct_split_pseudobulk.h5ad')
        b=ad.read_h5ad(dest/'batch_role_split_controls.h5ad')
        for obj in [a,b]:
            assert sparse.isspmatrix_csr(obj.X) and obj.X.dtype.kind in 'iu'
            assert (obj.X.data>0).all()
            assert np.array_equal(np.asarray(obj.X.sum(1)).ravel(),obj.obs.total_umis)
        assert a.var.equals(b.var)
        counts=f.groupby(['construct','split'],observed=True).size()
        assert [int(counts.get((s,int(r)),0)) for s,r in zip(a.obs.construct,a.obs.split,strict=True)]==a.obs.n_cells.tolist()
        for role in ['D_training','null_evaluation','effect_baseline']:
            labels=set(f.loc[f.control_role.eq(role),'construct'])
            assert (a.X[a.obs.construct.isin(labels).to_numpy()].sum(0)==b.X[b.obs.control_role.eq(role).to_numpy()].sum(0)).all()
        refpath=DATA/'input'/config['contexts'][c]['reference']
        assert sha(refpath)==config['contexts'][c]['reference_sha256']
        ref=ad.read_h5ad(refpath)
        assert a.var.equals(ref.var)
        groups=a.obs.groupby('construct',observed=True).indices
        exact=0
        for i,label in enumerate(ref.obs.guide_target):
            if label in groups:
                assert np.array_equal(np.asarray(a.X[groups[label]].sum(0)).ravel(),ref.X[i].toarray().ravel()),label
                exact+=1
        retained=[]
        for name in sorted(files):
            if name.startswith('responder_cells/') and not core_only:
                obj=ad.read_h5ad(dest/name,backed='r')
                assert obj.var.equals(a.var)
                retained.extend(obj.obs.locator.tolist())
                obj.file.close()
        if not core_only:
            assert len(retained)==len(set(retained))==acceptance['retained_cells']
            assert set(retained)==set(f.loc[f.retain_cell,'locator'])
        else:
            assert int(f.retain_cell.sum())==acceptance['retained_cells']
        support=pd.read_parquet(dest/'matched_control_support.parquet')
        missing=support[~support.matched_baseline_available]
        perts=missing[missing.control_role.eq('perturbation')]
        artifact_bytes=sum(d['bytes'] for d in files.values())+entry['bytes']
        result['contexts'][c]={'selected_cells':len(f),'retained_cells':acceptance['retained_cells'],'local_responder_cells_verified':len(retained),'artifact_bytes':artifact_bytes,'core_bytes':sum(d['bytes'] for n,d in files.items() if not n.startswith('responder_cells/'))+entry['bytes'],'responder_bytes':sum(d['bytes'] for n,d in files.items() if n.startswith('responder_cells/')),'files_verified':sum(not (core_only and n.startswith('responder_cells/')) for n in files)+1,'reference_constructs_exact':exact,'missing_baseline_strata':len(missing),'perturbation_constructs_with_missing_baseline':int(perts.construct.nunique()),'perturbation_targets_with_missing_baseline':int(perts.target.nunique()),'transport_audit_sha256':entry['sha256'],'files':files}
        result['total_artifact_bytes']+=artifact_bytes
        result['total_local_verified_bytes']+=sum(d['bytes'] for d in files.values() if d['verified_locally'])+entry['bytes']
        result['total_retained_cells']+=acceptance['retained_cells']
        print(json.dumps({c:{k:v for k,v in result['contexts'][c].items() if k!='files'}}),flush=True)
    (REPORT/('local-core-acceptance.json' if core_only else 'local-acceptance.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('All downloaded artifacts passed local hash and reconstruction checks.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only',action='store_true')
    main(parser.parse_args().core_only)
