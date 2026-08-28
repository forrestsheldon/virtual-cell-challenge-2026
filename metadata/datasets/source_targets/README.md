# Curated source target lists

These small text files contain source labels, one per line. They are inputs—not separate evidence—to `build_target_coverage.py`.

| file(s) | derivation | transfer |
|---|---|---:|
| `replogle_*.txt`, `nadig_*.txt` | `obs/perturbation/categories` read from scPerturb Zenodo H5ADs by HTTP range | 6.9-12.1 MB per file |
| `replogle_*_counts.csv`, `nadig_*_counts.csv` | category codes counted from the same remote H5AD metadata | same bounded read as target list |
| `vcc2025_h1.txt`, `vcc2025_h1_counts.csv` | categorical labels and codes from scPertEval `arch1` H5AD | 3.7 MB |
| `jiang_all.txt` | unique `Target Gene` values from author Supplementary Table 1f | source XLSX 3.0 MB |
| `jiang_*_counts.csv` | `meta.data$gene` counts by `cell_type`, summed across five checksum-verified author Seurat pathway objects | five temporary inputs totaling 20,141,612,637 bytes |
| `mcfaline_kinome.txt` | gene prefix from author GEO `GSM7056149_sciPlexGxE_2_gRNA_sequences.txt.gz`; header/control excluded | small compressed text |
| `mcfaline_kinome_guides.csv` | number of distinct reported guide sequences per included kinase target | same small compressed text |
| `mcfaline_*_counts.csv` | exact single-gene `colData$gene_id` counts by cell line from the author preprocessed CDS list | temporary GEO input 3,619,987,565 bytes |
| `jiang_cell_count_audit.csv`, `mcfaline_cell_count_audit.csv` | represented-cell, control/missing, multiple-assignment, and counted-single-target totals from extraction | generated metadata only |

Remote H5AD sizes range from 0.85 to 8.8 GB, but the expression arrays were not downloaded. The range reader has a default 95 MiB hard transfer budget per invocation.

The RPE1 and Nadig essential-panel lists have zero case-exact overlap with the current VCC 300; this was retained rather than “fixed” by speculative aliases. Their biology can still teach cross-gene context transfer. Jiang's same five pathway pools were applied to all six cell lines, and McFaline-Figueroa's 522-kinase pool was applied to the three coverage lines, so reusing those target files across their respective contexts reflects the experimental design.

The range reader has a default 95 MiB hard transfer budget per invocation and can
optionally emit category counts with `--codes`. Counts describe reported metadata
groups only; they are not quality filters. Control labels are retained in these
curated source files/counts for provenance and excluded only when constructing the
gene x dataset-context observation table. `KNTC1` is a real gene and is not treated
as a control label.

Jiang and McFaline author objects were downloaded to a temporary directory with
explicit approval, reduced with `scripts/metadata/extract_author_cell_counts.R`,
and deleted after count validation. Jiang has no multiple-gene labels in
`meta.data$gene`. McFaline comma-separated multiple-gene labels are reported in
the audit and excluded from the atlas's single-gene observation counts; no other
cell filtering was introduced.
