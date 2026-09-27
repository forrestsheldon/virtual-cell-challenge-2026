"""Compare compact split artifacts and count-ranked X-Atlas panels, using local metadata."""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/context-transfer-blog'
META = ROOT / 'data/derived/context_transfer_preflight/metadata'
GIB = 1024**3
CONTEXTS = ['K562_essential', 'RPE1', 'HepG2', 'Jurkat', 'HCT116', 'HEK293T']


def quant(values):
    return {str(q): float(np.quantile(values, q)) for q in [0, .25, .5, .75, 1]}


def ranking(paired):
    """Rank genes by their second-best shared construct's weaker-context cell count."""
    d = paired.reset_index()
    d['weaker_context_cells'] = d[['HCT116', 'HEK293T']].min(axis=1)
    d['both_contexts_cells'] = d[['HCT116', 'HEK293T']].sum(axis=1)
    d = d.sort_values(['target', 'weaker_context_cells', 'both_contexts_cells', 'construct'], ascending=[True, False, False, True])
    d['construct_rank'] = d.groupby('target').cumcount() + 1
    eligible = d.groupby('target').filter(lambda x: len(x) >= 2)
    out = eligible[eligible.construct_rank == 2].set_index('target')[['weaker_context_cells']].rename(columns={'weaker_context_cells':'pair_bottleneck_cells'})
    out['total_cells_both_contexts'] = eligible.groupby('target').both_contexts_cells.sum()
    out['shared_constructs'] = eligible.groupby('target').size()
    out['best_two_constructs'] = eligible[eligible.construct_rank <= 2].groupby('target').construct.agg(lambda x: json.dumps(list(x)))
    return out.reset_index().sort_values(['pair_bottleneck_cells', 'total_cells_both_contexts', 'target'], ascending=[False,False,True]).reset_index(drop=True)


def retain_responder_cells(r):
    """Restore the requested cell subsets using the archived metadata estimates."""
    old = json.loads((REPORT/'archive/split-artifact-preflight-v1/split-artifact-preflight.json').read_text())
    contexts = {}
    for pair, option in [('K562_essential__RPE1', 'all_shared'), ('HepG2__Jurkat', 'all_shared'), ('HCT116__HEK293T', 'union415')]:
        for c, d in old['pairs'][pair]['options'][option].items():
            contexts[c] = {k: v for k, v in d.items() if k.startswith('responder_')}
    total = sum(d['responder_CSR_GiB_all_controls'] for d in contexts.values())
    r['responder_retention'] = {
        'decision': 'Requested by Forrest 2026-09-24; retain for research expected within two weeks, with no automatic expiry. Analysis remains deferred.',
        'scope': '115 shared multi-construct targets in K562/RPE1, 135 in HepG2/Jurkat, and 17 in the original X-Atlas union; all shared constructs and all eligible controls from relevant batches. Added X-Atlas genes remain pseudobulk-only.',
        'format': 'Raw integer CSR counts on full native gene axes, canonical cell locators, batch/construct labels, source metadata, control roles and split assignments. No inferred responder labels or responsibilities.',
        'contexts': contexts,
        'CSR_GiB_estimate': total,
        'retention': 'Durable versioned derived objects; no automatic two-week deletion. User approval required for later deletion. Prefer cloud retention and local compact core/audits.'}
    for profile in r['storage_profiles'].values():
        profile['responder_CSR_GiB_estimate'] = total
        profile['total_retained_GiB_estimate'] = profile['core_GiB_budget'] + total
    r['artifact_contract_proposal']['responder_retention'] = r['responder_retention']['scope'] + ' Retain raw cell objects; responder analysis deferred. No automatic expiry.'
    r['artifact_contract_proposal']['batch_folds'] = 'Whole-batch folds remain optional for the core; retained responder cells permit later reaggregation only within their smaller panel.'
    r['storage_estimation']['responder'] = 'All eligible controls retained. Archived metadata estimates restored: scPerturb upstream ngenes; X-Atlas source-average density with 0.5x–2x uncertainty. CSR estimates exclude container overhead.'
    r['cloud_proposal']['retention'] += ' Retain responder-ready derived cell objects durably, without automatic expiry; local copy optional.'


def revise(r, frames):
    panel = set(r['panel_targets'])
    pooled = {}
    for c,f in frames.items():
        pooled[c] = f.groupby(['target','construct'], observed=True).n_cells.sum()
    paired = pd.concat([pooled[c].rename(c) for c in CONTEXTS[-2:]], axis=1).dropna()
    paired = paired[paired.index.get_level_values('target').str.lower() != 'non-targeting'].astype('int64')
    rank = ranking(paired)
    assert len(rank) == 1491
    rank['bottleneck_rank'] = np.arange(1,len(rank)+1)
    total_order = rank.sort_values(['total_cells_both_contexts','target'], ascending=[False,True])
    rank['total_count_rank'] = rank.target.map(dict(zip(total_order.target,range(1,len(rank)+1),strict=True)))
    rank['in_union415'] = rank.target.isin(panel)
    rank.to_csv(REPORT/'split-artifact-xatlas-ranked-genes.csv', index=False)
    for n in [100, 200]:
        rank.head(n).to_csv(REPORT/f'split-artifact-xatlas-top{n}.csv', index=False)

    # Controls are pooled by identity x split and, separately, by batch x role x split.
    supports = {}
    control_batches = {}
    for c in CONTEXTS:
        family = 'xatlas_orion' if c in CONTEXTS[-2:] else 'context_atlas'
        base = ROOT/f'data/derived/{family}/final/{c}'
        p = base/f'{c}_guide_pseudobulk.h5ad'
        a = ad.read_h5ad(p, backed='r')
        with h5py.File(p,'r') as h:
            supports[c] = pd.Series(np.diff(h['X/indptr'][...]),index=a.obs.guide_target.astype(str))
        a.file.close()
        if c in CONTEXTS[-2:]:
            p=base/f'{c}_batch_control_pseudobulk.h5ad'
            a=ad.read_h5ad(p,backed='r')
            with h5py.File(p,'r') as h:
                control_batches[c]=pd.Series(np.diff(h['X/indptr'][...]),index=a.obs['sample'].astype(str))
            a.file.close()

    shared_sc = {}
    for l,rr in [('K562_essential','RPE1'),('HepG2','Jurkat')]:
        shared_sc[l] = shared_sc[rr] = set(pooled[l].index.get_level_values('construct')) & set(pooled[rr].index.get_level_values('construct'))
    estimates = {}
    panel_stats = {}
    for method in ['bottleneck','total_count']:
        order = rank.sort_values(method+'_rank')
        for n in [0,100,200]:
            if method=='total_count' and n==0:
                continue
            selected_targets = set(order.head(n).target)
            name = f'union415_plus_{method}_top{n}' if n else 'union415'
            selected_x = set(paired.loc[paired.index.get_level_values('target').isin(panel|selected_targets)].index.get_level_values('construct'))
            detail = {}
            for c in CONTEXTS:
                f = frames[c]
                control = f[f.target.str.lower()=='non-targeting']
                allowed = selected_x if c in CONTEXTS[-2:] else shared_sc[c]
                p = f[f.construct.isin(allowed) & (f.target.str.lower()!='non-targeting')]
                counts = p.groupby('construct',observed=True).n_cells.sum()
                g = r['contexts'][c]['native_genes']; b=f.batch.nunique()
                labels = set(p.construct)|set(control.construct)
                # A partition cannot introduce a gene absent from its pooled row.
                # New X-Atlas genes lack local expression support: use full native axis.
                pooled_nnz_upper = int(supports[c].reindex(sorted(labels),fill_value=g).sum())
                expression_upper = 4*pooled_nnz_upper
                if 'nnz' in f:
                    expression_upper = min(expression_upper, int(p.nnz.sum()+control.nnz.sum()))
                    batch_upper = min(12*b*g,int(control.nnz.sum()))
                else:
                    batch_upper = min(12*b*g,12*int(control_batches[c].sum()))
                rows = 4*len(labels); batch_rows=12*b
                expression_bytes=12*expression_upper+8*(rows+1)
                batch_bytes=12*batch_upper+8*(batch_rows+1)
                selected_cells=int(p.n_cells.sum()+control.n_cells.sum())
                assignment_bytes=160*selected_cells
                # Fine-stratum counts/exposures/IDs only, without a gene vector per row.
                exposure_bytes=80*4*(len(p)+len(control))
                detail[c]={'targets':int(p.target.nunique()),'constructs':int(p.construct.nunique()),'perturbation_cells':int(p.n_cells.sum()),'selected_cells_including_controls':selected_cells,'pooled_construct_cell_quantiles':quant(counts),'quarter_cell_budget_quantiles':quant(counts/4),'construct_split_rows_including_controls_upper':rows,'batch_role_split_rows_upper':int(batch_rows),'construct_split_CSR_GiB_upper':expression_bytes/GIB,'batch_role_split_CSR_GiB_upper':batch_bytes/GIB,'assignment_GiB_estimate':assignment_bytes/GIB,'exposure_table_GiB_estimate':exposure_bytes/GIB,'core_GiB_budget':(expression_bytes+batch_bytes+assignment_bytes+exposure_bytes)/GIB}
            estimates[name]={'contexts':detail,'core_GiB_budget':sum(d['core_GiB_budget'] for d in detail.values())}
            if n:
                picked=order.head(n)
                panel_stats[name]={'ranking':method,'panel_size':n,'selected_targets':sorted(selected_targets),'already_in_union415':len(selected_targets & panel),'new_targets':len(selected_targets-panel),'pair_bottleneck_cells_min':int(picked.pair_bottleneck_cells.min()),'pair_bottleneck_cells_median':float(picked.pair_bottleneck_cells.median()),'pair_bottleneck_quarter_budget_min':float(picked.pair_bottleneck_cells.min()/4),'overlap_with_other_ranking':len(selected_targets & set(rank.sort_values(('total_count' if method=='bottleneck' else 'bottleneck')+'_rank').head(n).target))}
    r['revision']='v4: responder-ready cell retention requested; responder analysis deferred'
    # Earlier large-storage scenarios remain in the archived v1 report, not the current contract.
    for p in r['pairs'].values():
        for options in p['options'].values():
            for d in options.values():
                for k in list(d):
                    if k.startswith(('pseudobulk_', 'atomic_rows_', 'pooled_construct_split_rows_', 'assignment_uncompressed_', 'responder_')):
                        del d[k]
    r['xatlas_ranked_panels']=panel_stats
    r['xatlas_ranking_decision']={'criterion':'weaker of two best shared constructs across both contexts','selected_by':'Forrest','date':'2026-09-24','panel_size':'pending 100 versus 200; no build approved','total_count_comparison':'sensitivity only, not the selected method'}
    r['storage_profiles']=estimates
    r['artifact_contract_proposal']={
        'core':['selected-cell assignment metadata','construct x four cell-split raw pseudobulks including control identity x split','batch x frozen control-role x split raw control pseudobulks','construct/control identity x batch x split counts and UMI exposure metadata without expression','manifests and audits'],
        'not_in_core':['construct x batch x split expression','control identity x batch x split expression','responder single-cell counts','whole-batch fold expression'],
        'universe':'Broad exact shared constructs in scPerturb pairs. X-Atlas retains the 415-target panel; compare optional 100/200 count-ranked multi-construct genes as pseudobulks only. Retain every shared construct for selected genes, not only the two used to rank.',
        'ranking':'Chosen by Forrest 2026-09-24: for each shared construct take min(HCT116 cells, HEK293T cells); rank target by second-largest such count, then total cells across both contexts, then target label. Compare total-cell ranking explicitly. Selection is expression-blind but enriches well-sampled surviving perturbations; not a representative target sample.',
        'cell_splits':'Seed 0; SHA-256 order within context x source batch x exact construct; tie-break by canonical locator; round-robin four labels with hash-derived stratum rotation. Sum expression across batches. No per-batch n>=4 filter. Preserve empty-stratum counts; gate on pooled split support.',
        'cell_locator':'source SHA-256 + original file/shard + zero-based source row; retain source barcode/index',
        'exact_construct_id':'Canonical JSON [paired-library namespace, verbatim label]; no component reorder, alias remapping, or interpretation of P1/P2/P1P2',
        'controls':'Freeze identity-disjoint D-training, null-evaluation and effect-baseline pools; same identity role across contexts. Store both identity x split and batch x role x split sums, intentionally overlapping views of the same cells. Their marginal sums must reconstruct the same controls. Matched baseline excludes D/null pools and any null identity itself.',
        'batch_matching':'Use per-construct batch/split UMI totals to weight batch x baseline-role x split rates. Missing role/batch/split controls must be flagged; a supported-subset perturbation effect cannot be reconstructed by dropping unavailable batches after pooling. Stop the matched calculation or build an explicitly prespecified alternative during acquisition. Do not silently pool across batches.',
        'interpretation':'Cell-split reliability conditional on recorded experiment/batch mixture. Shared systematic batch effects can reproduce; no biological-replicate claim or arbitrary batch reweighting/bootstrap from core artifacts.',
        'batch_folds':'Optional extension, predefined whole-batch folds with domain-separated hash independent of cell splits. Core assignments retain source batch; fold aggregation or held-out responder modeling needs separately approved expression retention.',
        'responder_retention':'Populated below from the requested retained panel.',
        'source_preservation':'Raw nonnegative integer counts, full native gene axis, total UMI/cell counts; preserve upstream core_scale_factor verbatim where present without calling it a validated exposure. No normalization/filtering/HVG selection. Duplicate symbols collapsed only later in raw count space.',
        'logical_hashes':'Canonical gene order, sorted row keys, metadata and CSR content hashes independent of compression/time; separate file SHA-256 for transport integrity.',
        'staging_local':'data/derived/context_transfer_splits/<release>/<context>/',
        'staging_cloud':'gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/<release>/<context>/'
    }
    r['unresolved_decisions']=['X-Atlas augmentation: none, 100 or 200. Weaker-construct ranking chosen by Forrest; panel size and build not approved.','Control-role allocation and prespecified handling of missing batch/role/split controls.','Optional whole-batch validation folds; not required for initial cell-split reliability.','Approve source downloads, worker, disk, spend/retention lifecycle after revised proposal.']
    r['storage_estimation']={'expression':'Uncompressed int64 CSR data + int32 indices + int64 pointers; upper bounds from 4 x observed pooled gene support, capped by input nnz where known. New X-Atlas constructs without expression support use dense native-axis ceilings; no compression assumed.','metadata':'160 bytes per selected cell; 80 bytes per construct/control x batch x split exposure row. Planning estimates, not physical file measurements.','responder':'Populated below from archived metadata estimates for the retained panel.','exclusions':'H5AD/Parquet container metadata and temporary build copies additional; 50% working margin in disk budget.'}
    r['cloud_proposal']={'project':'bold-bastion-509200-f9','configuration':'vcc-2026','region':'europe-west2','zone':'europe-west2-b','machine':'e2-highmem-8','vcpus':8,'RAM_GiB':64,'boot_disk':{'GiB':30,'class':'pd-balanced'},'scratch_disk':{'GiB':100,'class':'pd-balanced'},'peak_memory_estimate_GiB':[8,24],'runtime_estimate_hours':[4,12],'pricing_status':'Prior Phase A planning allowances only; confirm London quote before creation. No new cloud operations in v2.','compute_USD_hour_allowance':.50,'balanced_disk_USD_GiB_month_allowance':.15,'storage_USD_GiB_month_allowance':.03,'ephemeral_IP_USD_hour_allowance':.005,'VM_disks_IP_USD_hour_allowance':.50+130*.15/730+.005,'retained_disks_USD_month_allowance':19.5,'source_downloads_GiB':(4927873119+126259821037)/GIB,'source_scan':'Both 100 and 200 panels still require the 117.59 GiB X-Atlas expression scan under full-shard SHA verification. Changes reduce output/storage/aggregation, not source acquisition.','execution':'One context and one source shard at a time; bounded count blocks; no fine-stratum expression accumulator. Stop compute on completion/safe failure.','retention':'Versioned staging only; independently validate local/cloud hashes and reconstruction. No persistent source mirror. Source/scratch deletion only under agreed lifecycle after validation. Canonical compact artifacts immutable.','service_account':'vcc-worker@bold-bastion-509200-f9.iam.gserviceaccount.com','resources_created':[],'cloud_writes':[],'local_disk_free_GiB_prior_observation':50,'pricing_sources':['https://cloud.google.com/products/compute/pricing/general-purpose','https://cloud.google.com/compute/disks-image-pricing','https://cloud.google.com/storage/pricing']}
    r['limitations']=['High-count ranking enriches for coverage/survival; failure of low-count genes is not adjudicated.','Pooled cell splits cannot identify biological versus reproducible batch effects.','Four nonempty cells per source batch are not required; matched control availability is a separate check.','Core artifacts cannot reconstruct responders, arbitrary batch bootstrap, or new perturbation batch weighting.','Prior source/hash checks remain dated 2026-09-24; no new expression acquisition or resource mutation.']
    r['supersedes']='archive/split-artifact-preflight-v1/; fine-stratum expression and associated 48–744 GiB estimates are not the current proposal'
    retain_responder_cells(r)
    (REPORT/'split-artifact-preflight.json').write_text(json.dumps(r,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x))+'\n')
    print(json.dumps({'storage':{k:round(v['core_GiB_budget'],3) for k,v in estimates.items()},'panels':{k:{a:b for a,b in v.items() if a!='selected_targets'} for k,v in panel_stats.items()}},indent=2))
    write_report(r)
    return r


def write_report(r):
    """Render the current proposal; all numeric tables come from the same JSON."""
    profiles=r['storage_profiles']
    names=['union415','union415_plus_bottleneck_top100','union415_plus_bottleneck_top200']
    lines=["# Cell-split artifact preflight — compact revision", "",
        "**Revised 2026-09-24. Proposal only; no source acquisition or cloud build authorized.**", "",
        "This revision replaces fine-batch expression with pooled construct splits. Broad scPerturb remains the working scope; X-Atlas keeps the 415-target union with a proposed 100- or 200-gene addition. H1 is unchanged. KOLF is excluded. The original preflight and its hashes are preserved in [archive/split-artifact-preflight-v1](archive/split-artifact-preflight-v1/README.md). Its 48–744 GiB storage scenarios are superseded, not current requirements.", "",
        "## Compact core", "",
        "- Raw **construct × four cell-split pseudobulks**, including control identity × split rows, pooled across source batches.",
        "- Raw **batch × frozen control-role × split control pseudobulks** for matching baselines.",
        "- Selected-cell assignments and **counts/UMI exposures** by construct or control identity × batch × split, without a gene-expression vector for each fine stratum.",
        "- Source manifests, canonical logical hashes, file SHA-256s and reconstruction audits.", "",
        "No construct × batch × split or control identity × batch × split expression is required. Responder analysis is deferred, but raw responder-ready cell subsets are retained separately at Forrest’s request; whole-batch-fold expression remains a separate possible extension. Preserve native gene axes, raw integer counts, source labels and encoded P-classes without normalization or promoter reinterpretation. Details: [artifact proposal](../../plans/context-transfer-split-artifacts.md) and [analysis plan](../../plans/context-transfer-split-power.md).", "",
        "## X-Atlas panel: weaker-construct criterion selected by Forrest", "",
        "For each exact shared construct, take min(HCT116 cells, HEK293T cells). Rank each eligible gene by its **second-best** construct on that measure; break ties by total cells across all shared constructs in both contexts, then gene label. There are 1,491 eligible multi-construct genes. Retain all shared constructs for selected genes, not just the two used for ranking. This favors well-sampled surviving perturbations and does not represent an unbiased genome-wide sample.", "",
        "| Added panel | New genes beyond union | Resulting X-Atlas targets / constructs per context | Minimum qualifying-pair cells, each context | Approx. quarter / half budget at that minimum | Core, all six contexts |",
        "|---|---:|---:|---:|---:|---:|"]
    for n in [100,200]:
        key=f'union415_plus_bottleneck_top{n}'
        p=r['xatlas_ranked_panels'][key]; d=profiles[key]['contexts']['HCT116']
        lines.append(f"| Top {n} | {p['new_targets']} | {d['targets']} / {d['constructs']} | {p['pair_bottleneck_cells_min']} | {p['pair_bottleneck_cells_min']/4:.1f} / {p['pair_bottleneck_cells_min']/2:.1f} | {profiles[key]['core_GiB_budget']:.2f} GiB |")
    lines += ["", "The count guarantee applies to the two qualifying constructs in each context, not weaker additional constructs. Quarter/half figures divide pooled counts; final batch-aware assignments need not have perfectly equal pooled sizes. Top100 is nested within top200. The original X-Atlas panel covers 414 of the 415 union targets and contains 431 perturbation constructs per context.", "",
        "**Recommendation:** top200 is a reasonable next proposal: it costs about 0.73 GiB more than top100 under conservative core budgets while preserving at least 179 cells per qualifying construct per context. Panel size still needs approval. [Top100 list](split-artifact-xatlas-top100.csv), [top200 list](split-artifact-xatlas-top200.csv), [all rankings](split-artifact-xatlas-ranked-genes.csv).", "",
        "For comparison, total-cell ranking gives a minimum qualifying-pair count of only 8 in both the top100 and top200 lists; it can hide a weak partner behind one abundant construct. It overlaps the chosen lists by 53/100 and 124/200 genes. It is retained as a comparison, not the selected criterion.", "",
        "## Revised storage", "",
        "All core figures below include all four scPerturb contexts at broad shared-construct scope plus both X-Atlas contexts. Expression figures are **uncompressed CSR upper bounds**, using int64 counts, int32 column indices and int64 row pointers. Metadata sizes are estimates; container metadata and working copies are additional. No compression savings are assumed.", "",
        "| X-Atlas scope | Core budget | Retained cell CSR estimate | Total estimate |",
        "|---|---:|---:|---:|"]
    for name in names:
        p=profiles[name]
        label={'union415':'Current union','union415_plus_bottleneck_top100':'Union + top100','union415_plus_bottleneck_top200':'Union + top200'}[name]
        lines.append(f"| {label} | {p['core_GiB_budget']:.2f} GiB | {p['responder_CSR_GiB_estimate']:.2f} GiB | {p['total_retained_GiB_estimate']:.2f} GiB |")
    lines += ["", "Responder analysis remains deferred; **retain raw single-cell subsets** for the 115/135 broad scPerturb multi-construct targets and the original 17 X-Atlas multi-construct targets, with all eligible controls from their batches. Added X-Atlas genes remain pseudobulk-only. This adds **31.71 GiB** of uncompressed CSR estimates, with X-Atlas density uncertain by 0.5x–2x. Preserve native genes, cell locators, source metadata and split assignments; validate subset counts and reconstructed aggregates independently. These are future build outputs, not objects already produced. Retain durably for the anticipated research within two weeks, with **no automatic expiry**; later deletion requires user approval. Iterative responsibilities/DE and held-out calibration remain future development.", "",
        "Core expression bounds use a partition property: partitions cannot introduce genes absent from the existing pooled row, so sum split nnz is at most four times pooled nnz; cap by input nnz where known. New X-Atlas constructs have no local pooled expression reference and use the full native-gene ceiling. Batch-role controls use the same support bound. Metadata allowances are 160 bytes per selected cell and 80 bytes per fine exposure row, with dictionary-encoded identifiers expected. Budget working margin rather than treating these as measured compressed file sizes.", "",
        "| Context, with union + top200 | Targets / constructs | Selected cells incl. controls | Median pooled construct cells | Core budget |",
        "|---|---:|---:|---:|---:|"]
    for c,d in profiles[names[-1]]['contexts'].items():
        lines.append(f"| {c} | {d['targets']:,} / {d['constructs']:,} | {d['selected_cells_including_controls']:,} | {d['pooled_construct_cell_quantiles']['0.5']:g} | {d['core_GiB_budget']:.2f} GiB |")
    lines += ["", "## What batch information still does", "",
        "Replogle batch labels were renamed from gem_group. Nadig supplies obs/batch; its experimental meaning remains unverified. X-Atlas sample labels correspond one-to-one with the 109 HCT116 and 223 HEK293T shards. These labels are not established biological replicates.", "",
        "Assign cells by deterministic hash order and round-robin within batch × exact construct, with a stratum-specific starting rotation; then pool expression across batches. **Do not require four cells in every batch or discard sparse strata.** The earlier occupancy measurements remain valid metadata, but n≥4 per batch is not an analysis gate. Gate on actual pooled split support. Record batch composition and exposures in the small tables.", "",
        "A matched baseline is the sum of batch × baseline-role × split control rates weighted by that construct split's batch UMI exposure fractions. It is computable from this core. It does not allow arbitrary later perturbation-batch reweighting. If a required control batch/split is missing, flag the estimate as unavailable; pooled perturbation expression cannot be retroactively restricted to the supported batches. Prespecify any alternative aggregate before acquisition rather than silently pooling controls or filtering cells. The illustrative identity-role allocation leaves one HepG2 batch with only three baseline controls; final support must be checked after assignment.", "",
        "Cell halves measure reliability within the recorded experiment/batch mixture. Shared systematic effects can reproduce, and different split mixtures can change the underlying effect. The off-diagonal product is shared split signal; interpretation as a single biological effect's power needs the corresponding assumptions and null checks. Whole-batch validation or arbitrary batch bootstrap requires additional approved expression retention. No such extension is silently promised by the compact core.", "",
        "## Preserved provenance and acquisition", "",
        "The initial Phase A checked all 14 retained H5AD hashes, all 332 X-Atlas shard source identities, every existing construct cell count reconstructed from source metadata, and scPerturb metadata UMI totals. The revision uses these cached metadata and local pooled-gene supports; it performs no new source/network acquisition. Full expression-file SHA-256s must still be verified after approved acquisition. Exact source sizes/URLs/MD5/SHA-256, axes, duplicate groups, control identities, batches and coverage quantiles remain in [the JSON](split-artifact-preflight.json).", "",
        "The four scPerturb sources are dense compressed float32 raw-count X, totaling 4,927,873,119 bytes (4.59 GiB), absent from repository data/external at preflight. Author processing and scPerturb harmonization are not independent experiments. No extra normalization is introduced. Replogle core_scale_factor is retained verbatim where present; capture-exposure semantics are unresolved.", "",
        "X-Atlas remains pinned to commit 53a5bc98d49247bcf967500292575c3d3602de31: 109 HCT116 shards (46,576,484,789 bytes) and 223 HEK293T shards (79,683,336,248 bytes), **117.59 GiB total**. Largest shard is 717,945,523 bytes. Under full-shard checksum validation, top100 and top200 require the same source scan. Smaller panels save output storage and aggregation work, not source download volume. Existing checkpoint sums cannot reconstruct cell splits or responders.", "",
        "Paired common native axes remain 7,226 / 7,632 / 38,606 gene IDs; X-Atlas has 38,584 literal symbols and 21 duplicate groups. Preserve all native columns; freeze any downstream mapping and target exclusion separately. The broader scPerturb pairs retain 115 / 135 multi-construct targets. H1 and the original 415-target union remain unchanged.", "",
        "## Revised cloud envelope — not authorization", "",
        "- Project bold-bastion-509200-f9, configuration vcc-2026, region europe-west2, proposed zone europe-west2-b; existing bucket-scoped worker service account and network. Verify boundary again before any mutation.",
        "- Candidate e2-highmem-8: 8 vCPU / 64 GiB; **30 GiB balanced boot + 100 GiB balanced scratch**, replacing the prior 300–2,000 GiB scratch proposals. One context/shard at a time, bounded count blocks. Estimated RAM 8–24 GiB; validate at the first-shard checkpoint. No dense fine-stratum accumulator.",
        "- Core plus retained cells totals about 41.57 GiB for top200, or 62.35 GiB with 50% working margin at central density. At twice the estimated X-Atlas cell density, the same margin reaches about 101 GiB before source/container overhead; the 100 GiB scratch candidate is conditional on bounded output staging and a measured first-shard checkpoint, otherwise revise disk size/cost before launch. Full sources must not accumulate on scratch: temporary source cleanup requires an agreed lifecycle and independently validated durable checkpoints.",
        "- Keep the conservative 4–12-hour runtime allowance until measured; reading/decompression and verification still dominate source acquisition. Prior pricing allowances (not a verified London quote): $0.50/VM-hour, $0.15/balanced-GiB-month, $0.005/address-hour, $0.03/durable-GiB-month. VM + disks + address are about **$0.532/hour**, or **$2.13–$6.38** over 4–12 hours. Confirm regional quote before creation; taxes, operations and transfer are additional.",
        "- If stopped and retained, both proposed disks cost about **$19.50/month** under those allowances. Core durable storage is about **$0.25–$0.30/month**; retained cells add **$0.95/month**, for about **$1.20–$1.25/month** total at central density. Archives/copies add cost. Local egress remains separate (prior allowance $0.15/GiB). These inherit the dated Phase A planning rates, not a new price verification.",
        "- Local free space was about 50 GiB at Phase A. Keep the compact core/audits locally and responder cell objects durably in the bucket; do not assume a full local copy fits with working margin. Recheck free space before the build. Local/cache statistics are dated, not a fresh whole-machine audit.",
        "- Stage only under data/derived/context_transfer_splits/<release>/<context>/ and gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/<release>/<context>/. Stop compute on completion/safe failure. Independently verify scientific reconstruction and local/cloud hashes before promotion; no canonical overwrite or persistent source mirror. Agree source/scratch retention/deletion before launch.", "",
        "## Acceptance and remaining decisions", "",
        "Reconstruct all included existing construct/control rows exactly. Control identity×split and batch×role×split are overlapping margins of the same cells and must reconstruct the same totals. Broad shared scPerturb omits some context-specific constructs: K562 EGLN2/PTCD1/RBM4 and RPE1 C7orf26/FAM136A/ZBTB17 require explicitly partial-target references. Added X-Atlas genes lack an existing expression reference, so require an independent unsplit source aggregation plus metadata counts; retain exact comparisons for every original union-panel row.", "",
        "Before launch: choose top100 versus top200 (recommend top200), finalize identity roles/missing-control handling, decide whether any batch-fold extension is needed, and approve acquisition/resources/spend/retention. No n≥4 batch filter is proposed. Establish per-perturbation signal/noise in raw total-UMI rate geometry first, then frozen-D calibration and construct agreement. Responder detection is not a current gate or deliverable.", "",
        "Reproduce cached reassessment with `.pixi/envs/default/bin/python scripts/analysis/reassess_split_storage.py`; the full metadata/local-hash audit script also emits this current revision. The ranking and support-bound calculations have synthetic positive/negative checks; this is not a claim that future builders or scientific gates have passed. [Current audit](split-artifact-preflight-audit.json).", "",
        "**Resource state for this revision:** no cloud commands, created resources, writes, source downloads, source deletion, commits or pushes. Project-wide resources/costs were not inspected; KOLF remains untouched. No new continuing cloud cost. Prior source evidence and cloud read-only checks remain dated 2026-09-24.", ""]
    (REPORT/'split-artifact-preflight.md').write_text('\n'.join(lines))


if __name__=='__main__':
    r=json.loads((REPORT/'archive/split-artifact-preflight-v1/split-artifact-preflight.json').read_text())
    frames={c:pd.concat([pd.read_parquet(p) for p in sorted((META/c).glob('*.coverage.parquet'))],ignore_index=True) if c in CONTEXTS[-2:] else pd.read_parquet(META/f'{c}_coverage.parquet') for c in CONTEXTS}
    revise(r,frames)
