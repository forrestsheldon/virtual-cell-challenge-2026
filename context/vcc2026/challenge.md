---
last_verified: 2026-08-21
status: current
---

# Prediction task

VCC 2026 is a **zero-shot, multi-context** prediction task. For each anonymous context, participants receive only non-targeting-control (NTC) single-cell expression plus a target list and must generate the post-CRISPRi expression distribution for every target. No perturbed cells from the 2026 target domains are supplied for model fitting.

There are six anonymized human cell lines from different tissues. Validation uses contexts A/B/C and one 300-gene panel; final evaluation uses unseen contexts D/E/F and a different 300-gene panel. Validation and final scores are therefore not directly comparable, and final evaluation alone determines rank.

For each round the required prediction contains:

- 3 contexts × 300 targets × 400 cells = **360,000 cells**;
- **18,533 genes**, in the provided order;
- `target_gene` and `context` for each predicted cell;
- no predicted control cells.

The task permits learning from VCC 2025 H1 and other public or lawfully usable private data, subject to the rules. The defining modeling problem is context transfer: learn perturbation effects elsewhere, then condition them on control expression from a new anonymous context.
