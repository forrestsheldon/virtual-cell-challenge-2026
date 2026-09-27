# Chromatin-accessibility context atlas

Accessed 2026-09-20. The local collection contains 10 source files spanning
nine contexts and occupies 463,227,825 bytes (441.77 MiB). All coordinates are
GRCh38/hg38. Exact URLs, source/processing provenance, checksums, and caveats
are frozen in
[`metadata/datasets/chromatin_accessibility_sources.json`](../../metadata/datasets/chromatin_accessibility_sources.json).

## What was collected

| Context | Assay and representation | Relationship to perturbation data |
|---|---|---|
| H1 | ENCODE DNase narrow peaks | Cell-line match; independent assay and experiment |
| K562 | ENCODE ATAC conservative IDR peaks | Cell-line match; independent experiment |
| HepG2 | ENCODE ATAC conservative IDR peaks | Cell-line match; independent experiment |
| HCT116 | ENCODE ATAC conservative IDR peaks | Cell-line match; independent experiment |
| CD4 T cell | ENCODE ATAC IDR peaks | Cell-type proxy; donor/state unmatched |
| HEK293T | ENCODE DNase narrow peaks | Cell-line match, but poor ENCODE QC; sensitivity only |
| Jurkat | Two ChIP-Atlas ATAC peak sets | Unstimulated cell-line match; uniform reprocessing of two SRA replicates |
| KOLF | Author ATAC optimal peaks | Engineered KOLF2.1J-derived iTF-iPSC proxy, not parental KOLF2.1J |
| RPE1 | Author ATAC BigWig signal | RB-WT asynchronous cell-line match; CPM signal, no deposited peaks |

The source files are intentionally untouched. There is no local peak merging,
liftover, promoter assignment, assay normalization, or binary accessibility
matrix yet.

## Reproduce and audit

```bash
pixi run python scripts/data/download_chromatin_accessibility.py
pixi run python scripts/data/audit_chromatin_accessibility.py
```

The audit verifies byte counts, MD5 and SHA-256 hashes, all BED interval
coordinates, record counts, gzip integrity through full decompression, and the
BigWig magic number. The completed audit has zero malformed intervals across
1,178,360 peak records.

## Modeling boundary

For a first multimodal regression extension, derive features separately before
combining them:

1. binary promoter accessibility from peak overlap;
2. peak distance or count in a wider cis window;
3. RPE1 mean signal in the same windows;
4. an assay/provenance indicator so DNase, ENCODE ATAC, ChIP-Atlas ATAC, and
   author processing are not treated as exchangeable measurements.

The first scientific test should be whether target-context accessibility adds
held-out explanatory power beyond control expression. It should not begin with
a high-dimensional peak-union model: context and processing source are partly
confounded in this small panel.

## Known limitations

- The accessibility measurements were not collected alongside the perturbation
  screens, so they are context covariates rather than matched multimodal data.
- H1 and HEK293T are DNase-seq. HEK293T additionally has an ENCODE
  `extremely low spot score` audit error.
- The KOLF proxy contains an engineered six-transcription-factor cassette at
  the CLYBL locus.
- The CD4 sample is a single unmatched primary donor.
- Jurkat peak calls come from ChIP-Atlas uniform reprocessing because the
  authors deposited TDF browser tracks rather than BED peaks.
- RPE1 has author CPM-normalized signal but no author peak set; it must not be
  silently binarized alongside the peak files.
