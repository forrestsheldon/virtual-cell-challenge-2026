"""Synthetic acceptance controls for raw split retention and independent audit."""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/cloud'))
from aggregate_scperturb_context import sha256
from audit_context_splits import audit
from build_context_splits import assign, build, canonical, control_role


def fixture(tmp_path, kind):
    inp = tmp_path/'input'; inp.mkdir()
    work = tmp_path/'work'; src = work/'sources'/'toy';src.mkdir(parents=True)
    controls = []
    for role in ['D_training','null_evaluation','effect_baseline']:
        controls.append(next(f'NTC{i}' for i in range(100) if control_role(f'NTC{i}') == role))
    labels = ['gA','gB',*controls]
    rows = [(b,s,i) for b in ['a','b'] for s in labels for i in range(5 if s!='gB' else 1)]
    obs = pd.DataFrame(rows,columns=['batch','guide_id','i'])
    obs['gene'] = ['target' if s.startswith('g') else 'non-targeting' for s in obs.guide_id]
    x = (np.arange(len(obs)*3).reshape(-1,3) % 11).astype('int64')
    x[::3,0] = 0
    obs['ncounts'] = x.sum(1);obs['ngenes'] = (x>0).sum(1);obs['core_scale_factor'] = 1.5
    obs.index = pd.Index([f'cell{i}' for i in range(len(obs))])
    var = pd.DataFrame({'gene_name':['one','two','two'],'ensembl_id':['id1','id2','id3']},index=['id1','id2','id3'])
    if kind == 'scperturb':
        source = src/'source.h5ad'; ad.AnnData(x.astype('float32'),obs,var).write_h5ad(source)
    else:
        var['gene_token_id'] = [0,1,2]
        source = src/'source.parquet'
        pq.write_table(pa.table({'gene_token_id': [[0,1,2]]*len(x),'gene_expression':x.astype(float).tolist(),'gene_target':obs.gene.tolist(),'guide_target':obs.guide_id.tolist(),'sample':obs.batch.tolist(),'cell_barcode':obs.index.tolist(),'pass_guide_filter':[1]*len(x),'total_counts':x.sum(1),'n_genes_by_counts':(x>0).sum(1)}),source)
    ref = ad.AnnData(sparse.csr_matrix(np.vstack([x[obs.guide_id.eq(s)].sum(0) for s in labels])),pd.DataFrame({'guide_target':labels,'gene_target':['target','target',*(['non-targeting']*3)],'n_cells':[int(obs.guide_id.eq(s).sum()) for s in labels]},index=labels),var)
    ref.write_h5ad(inp/'ref.h5ad')
    cov = obs.rename(columns={'guide_id':'construct','gene':'target'}).groupby(['construct','target','batch']).size().rename('n_cells').reset_index()
    cov.to_parquet(inp/'coverage.parquet',index=False)
    cfg = {'context':'toy','namespace':'toy-pair','kind':kind,'construct_targets':{'gA':'target','gB':'target'},'controls':controls,'batches':['a','b'],'responder_targets':['target'],'responder_batches':['a','b'],'sources':[{'name':source.name,'url':'unused','sha256':sha256(source),'bytes':source.stat().st_size}],'coverage':'coverage.parquet','reference':'ref.h5ad','reference_sha256':sha256(inp/'ref.h5ad'),'coverage_sha256':sha256(inp/'coverage.parquet')}
    (inp/'release.json').write_text(json.dumps({'contexts':{'toy':cfg}}))
    return inp,work,cfg


@pytest.mark.parametrize('kind',['scperturb','xatlas'])
def test_build_and_independent_source_audit(tmp_path,kind):
    inp,work,cfg = fixture(tmp_path,kind)
    build(inp,work,cfg);audit(inp,work,'toy')
    out=work/'outputs'/'toy'
    f=pd.read_parquet(out/'cell_assignments.parquet')
    assert f.construct.eq('gB').sum()==2  # sparse strata retained
    assert json.loads((out/'acceptance.json').read_text())['status']=='passed'
    # Corruption is rejected, even when file-level manifest hashes are updated.
    path=out/'construct_split_pseudobulk.h5ad'
    a=ad.read_h5ad(path);a.X.data[0]+=1;a.write_h5ad(path)
    m=json.loads((out/'manifest.json').read_text());m['files'][path.name]['sha256']=sha256(path);m['files'][path.name]['bytes']=path.stat().st_size
    (out/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(AssertionError):audit(inp,work,'toy')


def test_assignment_order_and_locator_identity():
    f=pd.DataFrame({'batch':['b']*13,'construct':['g']*13,'locator':[canonical(['sha','file',i]) for i in range(13)]})
    a=assign(f,'c').set_index('locator').split.sort_index()
    b=assign(f.sample(frac=1,random_state=2),'c').set_index('locator').split.sort_index()
    pd.testing.assert_series_equal(a,b)
    assert a.value_counts().max()-a.value_counts().min()==1


def test_raw_count_negative_control(tmp_path):
    inp,work,cfg=fixture(tmp_path,'scperturb')
    path=work/'sources'/'toy'/'source.h5ad'
    a=ad.read_h5ad(path);a.X[0,0]=.5;a.write_h5ad(path)
    cfg['sources'][0].update(sha256=sha256(path),bytes=path.stat().st_size)
    with pytest.raises(ValueError,match='Non-count'):build(inp,work,cfg)


def test_matched_baseline_and_shared_control_noise():
    # Two batches with different baseline rates: pooling introduces bias.
    rates=np.array([1.,9.]);weights=np.array([.9,.1])
    matched=weights@rates
    assert matched==pytest.approx(1.8)
    assert rates.mean()!=matched
    available=np.array([10,0])>0
    assert not available[weights>0].all()  # missing required batch unavailable
    rng=np.random.default_rng(0);n=100000
    p1,p2,c1,c2=rng.normal(size=(4,n))
    assert np.cov(p1-c1,p2-c1)[0,1]>.9
    assert abs(np.cov(p1-c1,p2-c2)[0,1])<.03
