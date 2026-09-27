"""Summarize immutable pseudobulks and metadata-only source coverage for Phase A."""
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'data/derived/context_transfer_preflight/metadata'
OUT = ROOT / 'reports/context-transfer-blog'
PAIRS = [('K562_essential','RPE1'), ('HepG2','Jurkat'), ('HCT116','HEK293T')]
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def quant(x):
    return {str(q):float(np.quantile(x,q)) for q in [0,.05,.25,.5,.75,.95,1]} if len(x) else {}


def pclass(s):
    return '|'.join(re.findall(r'(?:_|-)(P1P2|P1|P2)(?=-|\||$)', s)) or 'unencoded'


def occupancy(f):
    n = f.n_cells
    by = f.groupby('construct', observed=True)
    support = f[n>=4]
    return {'occupied_construct_batch_strata':len(f), 'cells':int(n.sum()), 'cells_per_stratum_quantiles':quant(n),
            'strata_ge4':int((n>=4).sum()), 'strata_ge8':int((n>=8).sum()), 'strata_ge20':int((n>=20).sum()),
            'cells_in_strata_ge4':int(support.n_cells.sum()), 'nonempty_atomic_rows':int(np.minimum(n,4).sum()),
            'constructs_with_any_stratum_ge4':support.construct.nunique(),
            'constructs_with_all_occupied_strata_ge4':int((by.n_cells.min()>=4).sum()),
            'constructs_with_at_least_two_batches_ge4':int((support.groupby('construct').batch.nunique()>=2).sum()),
            'pooled_construct_cell_quantiles':quant(by.n_cells.sum())}


panel = set(pd.read_csv(ROOT/'data/controls/pert_counts.csv').target_gene) | set(pd.read_csv(ROOT/'reports/vcc2026-h1/reference_cells.csv').target_gene)
assert len(panel)==415
R={'access_date':'2026-09-24','phase':'A; metadata and local immutable artifacts only','panel_targets':sorted(panel),'contexts':{},'pairs':{},'checks':{}}
frames={}
vars={}
for c in [c for pair in PAIRS for c in pair]:
    family='xatlas_orion' if c in PAIRS[-1] else 'context_atlas'
    root=ROOT/f'data/derived/{family}/final/{c}'
    manifest=json.loads((root/f'{c}_manifest.json').read_text())
    a=ad.read_h5ad(root/f'{c}_guide_pseudobulk.h5ad',backed='r')
    obs=a.obs.rename(columns={'guide_target':'construct','gene_target':'target'}).copy()
    obs['is_control']=obs.target.str.lower().eq('non-targeting')
    gene=a.var.copy(); gene['native_id']=a.var_names
    vars[c]=gene
    a.file.close()
    local_hashes={}
    for name,m in manifest['outputs'].items():
        p=root/name
        if p.exists():
            h=sha(p)
            assert h==m['sha256']
            local_hashes[name]=h
    if family=='xatlas_orion':
        records=[json.loads(p.read_text()) for p in sorted((META/c).glob('*.metadata.json'))]
        assert len(records)==manifest['source_shards'] and all(m['status']=='complete' for m in records), (c,len(records))
        parts=[]
        for p in sorted((META/c).glob('*.coverage.parquet')):
            q=pd.read_parquet(p); q['shard']=p.name.split('.coverage')[0]; parts.append(q)
        f=pd.concat(parts,ignore_index=True)
        shardman=[json.loads(p.read_text()) for p in sorted((META/c).glob('*_Batch*.json')) if '.metadata.' not in p.name]
        assert len(shardman)==manifest['source_shards']
        for sm in shardman:
            sub=f[f.shard==Path(sm['source_path']).stem]
            assert sub.batch.nunique()==1 and set(sub.batch)==set(sm['samples'])
            original=sub[sub.target.isin(panel)|sub.target.eq('Non-Targeting')]
            assert original.n_cells.sum()==sm['n_selected_cells']
        source={'revision':manifest['revision'],'shards':shardman,'bytes':sum(m['source_size'] for m in shardman),'metadata_bytes_read':sum(m['metadata_bytes_read'] for m in records),'schemas':sorted({m['schema'] for m in records}), 'n_cells':int(f.n_cells.sum()),'nnz_footer_upper':sum(e['values'] for m in records for e in m['expression_columns'] if e['name'].startswith('gene_expression'))}
        source['mean_nnz_per_source_cell_upper']=source['nnz_footer_upper']/source['n_cells']
        selected=f[f.target.isin(panel)|f.target.str.lower().eq('non-targeting')]
    else:
        f=pd.read_parquet(META/f'{c}_coverage.parquet')
        meta=json.loads((META/f'{c}_metadata.json').read_text())
        source={**{k:manifest[k] for k in ['source_url','source_file','source_size','source_md5','source_sha256']}, **meta,
                'local_availability':'No matching source filename found under repository data/external; remote worker storage not inspected',
                'acquisition':'Full verified source required for expression pass after approval; metadata only acquired in Phase A'}
        selected=f
    check=selected.groupby('construct',observed=True).n_cells.sum().sort_index()
    expected=obs.set_index('construct').n_cells.sort_index()
    pd.testing.assert_series_equal(check,expected,check_names=False,check_dtype=False)
    if 'total_umis' in f:
        assert int(f.total_umis.to_numpy().sum(dtype=np.int64))==manifest['selected_umis']
    f['is_control']=f.target.str.lower().eq('non-targeting')
    f['pclass']=f.construct.map(pclass)
    assert f.groupby('construct',observed=True).target.nunique().max()==1
    labels=f.construct.unique()
    assert all('|'.join(s.split('|'))==s and all(s.split('|')) for s in labels)
    source['construct_component_counts']=pd.Series([len(s.split('|')) for s in labels]).value_counts().sort_index().to_dict()
    frames[c]=f
    dup=gene[gene.gene_name.duplicated(False)].groupby('gene_name',observed=True).native_id.apply(list).to_dict()
    aliases=gene[gene.gene_name.str.match(r'^(TBCE|HSPA14)_')][['gene_name','native_id']].to_dict('records')
    R['contexts'][c]={'source':source,'existing_manifest':manifest,'local_output_hashes_verified':local_hashes,'native_genes':len(gene),'duplicate_symbol_groups':dup,'scperturb_disambiguated_symbols':aliases,'control_identities':sorted(f.loc[f.is_control,'construct'].unique()),'batches':sorted(f.batch.unique()),'all_perturbation_occupancy':occupancy(f[~f.is_control]),'control_occupancy':occupancy(f[f.is_control]),'control_cells_per_batch':f[f.is_control].groupby('batch').n_cells.sum().to_dict()}
    R['checks'][c]={'metadata_pooled_construct_counts_equal_existing':True,'native_local_file_hashes_match_manifest':True,'target_per_construct_unique':True,'component_label_roundtrip':True}

for l,r in PAIRS:
    lf,rf=frames[l],frames[r]
    lp,rp=lf[~lf.is_control],rf[~rf.is_control]
    shared=set(lp.construct)&set(rp.construct)
    targets=set(lp.target)&set(rp.target)
    assert lp[['construct','target']].drop_duplicates().merge(rp[['construct','target']].drop_duplicates(),on='construct').eval('target_x == target_y').all()
    gcommon=set(vars[l].gene_name)&set(vars[r].gene_name)
    idcommon=set(vars[l].native_id)&set(vars[r].native_id)
    key=f'{l}__{r}'
    pair={'shared_exact_constructs':len(shared),'shared_targets':len(targets),'shared_union415_targets':len(targets & panel),'shared_union415_constructs':len(shared & set(lp.loc[lp.target.isin(panel),'construct'])),'shared_controls':sorted(set(lf.loc[lf.is_control,'construct'])&set(rf.loc[rf.is_control,'construct'])),'common_response_symbols':sorted(gcommon),'common_native_gene_ids':len(idcommon),'options':{}}
    for name, constructs in [('union415',set(pd.concat([lp,rp]).loc[lambda d:d.target.isin(panel),'construct'])),('all_shared',shared)]:
        # Union415 retains every context's observed construct for those target genes;
        # all_shared retains only verbatim constructs observed in both contexts.
        o={}
        for c in [l,r]:
            f=frames[c]; p=f[~f.is_control & f.construct.isin(constructs)]
            ctrl=f[f.is_control]
            u=p[['construct','target','pclass']].drop_duplicates()
            mult=u.groupby('target').construct.nunique()
            mult=set(mult[mult>=2].index)
            mp=p[p.target.isin(mult)]
            batches=set(mp.batch)
            cc=ctrl[ctrl.batch.isin(batches)]
            # Illustrative cap: up to 128 controls per recorded batch, all eligible
            # identities first then deterministic balanced allocation within identity.
            subset=int(cc.groupby('batch').n_cells.sum().clip(upper=128).sum())
            g=R['contexts'][c]['native_genes']
            rows_all=int(np.minimum(p.n_cells,4).sum())
            rows_multi=int(np.minimum(mp.n_cells,4).sum())
            rows_ctrl=int(np.minimum(ctrl.n_cells,4).sum())
            pooled_rows=4*(p.construct.nunique()+ctrl.construct.nunique())
            if 'nnz' in p:
                resp_nnz=int(mp.nnz.sum()+cc.nnz.sum())
                resp_bytes=(12*resp_nnz+8*(int(mp.n_cells.sum()+cc.n_cells.sum())+1))/GIB
                estimate_kind='exact selected upstream ngenes sum; uncompressed CSR int64 data + int32 indices; verify ngenes against X in build'
            else:
                mean=R['contexts'][c]['source']['mean_nnz_per_source_cell_upper']
                resp_bytes=(12*mean*(mp.n_cells.sum()+cc.n_cells.sum())+8*(mp.n_cells.sum()+cc.n_cells.sum()+1))/GIB
                estimate_kind='source-average nnz from Parquet footers; selected density unknown; use 0.5x–2x planning range'
            ctrln=int(cc.n_cells.sum())
            sub_bytes=resp_bytes*(int(mp.n_cells.sum())+subset)/max(1,int(mp.n_cells.sum())+ctrln)
            mean = R['contexts'][c]['source'].get('mean_nnz_per_source_cell_upper')
            def nnz_est(frame, mean=mean):
                return float(frame.nnz.sum()) if 'nnz' in frame else float(mean*frame.n_cells.sum())
            bulk_pooled = min(nnz_est(p)+nnz_est(ctrl),pooled_rows*g)
            bulk_controls = min(nnz_est(ctrl),rows_ctrl*g)
            bulk_all = min(nnz_est(p),rows_all*g)
            bulk_multi = min(nnz_est(mp),rows_multi*g)
            bulk_all_gib=12*(bulk_pooled+bulk_controls+bulk_all)/GIB
            bulk_multi_gib=12*(bulk_pooled+bulk_controls+bulk_multi)/GIB
            o[c]={'pseudobulk_nnz_bound_or_estimate_GiB_all_batch':bulk_all_gib,'pseudobulk_nnz_bound_or_estimate_GiB_multiconstruct_batch':bulk_multi_gib,'pseudobulk_nnz_estimate_basis':'Upper bound from upstream ngenes (scPerturb); source-average density planning estimate (X-Atlas). CSR row pointers and H5AD metadata additional; gzip compression not assumed.', 'targets':p.target.nunique(),'constructs':p.construct.nunique(),'shared_constructs':len(set(p.construct)&shared),'perturbed_cells':int(p.n_cells.sum()),'control_cells':int(ctrl.n_cells.sum()),'selected_cells':int(p.n_cells.sum()+ctrl.n_cells.sum()),'target_list':sorted(p.target.unique()),'multiconstruct_targets':sorted(mult),'multi_pclass_targets':int((u.groupby('target').pclass.nunique()>1).sum()),'pclass_construct_counts':u.pclass.value_counts().to_dict(),'occupancy':occupancy(p),'multiconstruct_occupancy':occupancy(mp),'responder_perturbed_cells':int(mp.n_cells.sum()),'responder_batches':len(batches),'responder_all_controls':ctrln,'responder_subset_controls_cap128_per_batch':subset,'responder_CSR_GiB_all_controls':float(resp_bytes),'responder_CSR_GiB_subset_controls':float(sub_bytes),'responder_size_estimate_basis':estimate_kind,'responder_dense_CSR_upper_GiB':float(12*g*(mp.n_cells.sum()+cc.n_cells.sum())/GIB),'atomic_rows_all_constructs_nonempty':rows_all,'atomic_rows_multiconstruct_nonempty':rows_multi,'atomic_rows_control_identities_nonempty':rows_ctrl,'pooled_construct_split_rows_upper':pooled_rows,'pseudobulk_dense_CSR_upper_GiB_all_batch':12*g*(rows_all+rows_ctrl+pooled_rows)/GIB,'pseudobulk_dense_CSR_upper_GiB_multiconstruct_batch':12*g*(rows_multi+rows_ctrl+pooled_rows)/GIB,'assignment_uncompressed_MiB_estimate_160bytes_per_cell':160*(int(p.n_cells.sum()+ctrl.n_cells.sum()))/1024**2}
        pair['options'][name]=o
    R['pairs'][key]=pair

out=OUT/'split-artifact-preflight.json'
print('Wrote',out)
for k,p in R['pairs'].items():
    print(k,'shared constructs',p['shared_exact_constructs'],'targets',p['shared_targets'],'shared controls',len(p['shared_controls']),'common genes',len(p['common_response_symbols']))
    for name,o in p['options'].items():
        for c,v in o.items():
            print(name,c,{k:v[k] for k in ['targets','constructs','selected_cells','multi_pclass_targets','responder_perturbed_cells','responder_all_controls','responder_subset_controls_cap128_per_batch','responder_CSR_GiB_all_controls','responder_CSR_GiB_subset_controls','pseudobulk_dense_CSR_upper_GiB_all_batch','pseudobulk_dense_CSR_upper_GiB_multiconstruct_batch']})

# Design proposals and audit trail: these do not assign any source cells.
R['source_identity_checks'] = {
    'zenodo': json.loads((META/'zenodo-source-identity.json').read_text()),
    'xatlas_all_332_pinned_sizes_and_lfs_sha256_match': json.loads((META/'xatlas-pinned-source-identity.json').read_text())['all_332_sizes_and_lfs_hashes_match'],
    'full_source_sha256_recomputed_this_phase': False,
    'hash_provenance': 'scPerturb SHA-256 from prior audited build; MD5/size rechecked against Zenodo. X-Atlas SHA-256 rechecked against pinned LFS metadata. Full source bytes were not acquired.'}
for c in PAIRS[0]+PAIRS[1]:
    s=R['contexts'][c]['source']
    z=next(f for f in R['source_identity_checks']['zenodo']['files'] if f['key']==s['source_file'])
    assert z['size']==s['source_size'] and z['checksum']=='md5:'+s['source_md5']
    s['download_url']=z['links']['self']
for c,d in R['contexts'].items():
    f=frames[c]; ctr=f[f.is_control].copy()
    ctr['proposed_role']=ctr.construct.map(lambda s:['D_training','null_evaluation','effect_baseline','effect_baseline'][int.from_bytes(hashlib.sha256(('control-role-v1|'+s).encode()).digest()[:8],'big')%4])
    d['illustrative_control_role_counts']=ctr.groupby('proposed_role').agg(identities=('construct','nunique'),cells=('n_cells','sum')).to_dict('index')
    baseline=ctr[ctr.proposed_role=='effect_baseline'].groupby('batch').n_cells.sum().reindex(d['batches'],fill_value=0)
    d['illustrative_baseline_role_cells_per_batch_min']=int(baseline.min())
    d['illustrative_baseline_role_batches_under4_cells']=int((baseline<4).sum())
    d['metadata_file_sha256']={str(p.relative_to(ROOT)):sha(p) for p in ([META/f'{c}_coverage.parquet',META/f'{c}_metadata.json'] if c not in PAIRS[-1] else sorted((META/c).glob('*')))}
R['reconstruction_scope']={'K562_essential_partial_target_rows_all_shared':['EGLN2','PTCD1','RBM4'],'RPE1_partial_target_rows_all_shared':['C7orf26','FAM136A','ZBTB17'],'rule':'Compare included construct rows exactly; compare full target rows only where all source constructs for that target are included. Otherwise compare with exact sum of the included immutable reference construct rows, explicitly labelled partial-target universe. All control identities retained for aggregation, irrespective of responder subsampling.'}
# The compact v2 proposal is authoritative; do not regenerate the superseded v1 contract.
from reassess_split_storage import revise

R['validation'] = {**R.pop('checks'),
    'zenodo_sizes_md5_match': True,
    'xatlas_shard_sample_and_selected_counts_match_every_manifest': True,
    'cross_context_shared_construct_targets_match': True,
    'scperturb_metadata_UMI_totals_equal_existing': True}
revise(R, frames)
