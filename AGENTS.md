# Agent guidance

This repository is the source of truth for Virtual Cell Challenge 2026 analysis and modeling. Before assuming anything about the Challenge, read `context/vcc2026/`. Current official VCC sources override model memory and stale local notes; rules, evaluation, dates, and tooling may change during the competition.

- Preserve source URLs, access dates, versions, tags, and commits. Record unknowns explicitly.
- Distinguish original experiments, author-processed data, independent reprocessing/harmonization, mirrors, and convenience loaders. Two hosted copies of one processed dataset are not independent evidence.
- Inspect metadata and preprocessing provenance before selecting data. Do not download large datasets speculatively or any individual file over 100 MB without explicit approval.
- Prefer raw counts when analyses require our own biological or preprocessing choices. Processed copies can be valuable, but document their transformations first.
- Never silently normalize, log-transform, filter, batch-correct, or select HVGs. Never overwrite raw data.
- Keep large data and derived matrices gitignored. Make derived outputs reproducible from scripts and configuration where practical.
- Motivate modeling choices by the zero-shot structure of the 2026 task.
- Do not commit or push unless explicitly instructed.
