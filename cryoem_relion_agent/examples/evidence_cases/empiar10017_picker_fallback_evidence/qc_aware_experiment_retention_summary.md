# EMPIAR-10017 QC-aware Closed-Loop Selection Summary

## Purpose

This experiment evaluates whether CryoAgent can move beyond particle-count-based rerun selection and use ground-truth particle-picking quality metrics for closed-loop backend/configuration selection.

## Baseline

The baseline same-dataset workflow used EMPIAR-10017 full84:

```text
Topaz picking
→ RELION particle extraction
→ RELION reconstruction
→ map2pdb dry-run routing
→ rule-based decision trace
→ LLM-supervised audit

