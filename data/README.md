# Data

Challenge data live here locally and are ignored by git.

When the 2026 release is available, record:

- official source URLs and access date;
- filenames, sizes, and checksums;
- the distinction between raw, provided-preprocessed, and locally derived data;
- any commands used to download or transform the files.

Do not commit challenge data to this repository.

## External perturbation data

### Replogle 2022 K562 genome-wide Perturb-seq

- Local file: `external/replogle2022/ReplogleWeissman2022_K562_gwps.h5ad`
- Source: `https://zenodo.org/api/records/13350497/files/ReplogleWeissman2022_K562_gwps.h5ad/content`
- Accessed: 2026-08-21
- Upstream representation: scPerturb harmonization of the author-processed Replogle K562 GWPS experiment
- Exact size: 8,805,466,154 bytes
- MD5: `13db594f8f1d2ccb88fec44a13e414dc`
- Expression: 1,989,578 cells x 8,248 genes; dense `float32` raw UMI counts in gzip-compressed `X`
- Download method: resumable HTTP transfer, parallelized over non-overlapping byte ranges; ranges were reassembled in order and the complete file was checked against the Zenodo size and MD5 before the temporary chunks were removed.
