# CryoAgent Real-Run Evidence Cases

This directory contains curated evidence fixtures derived from real CryoAgent same-dataset runs.

The raw large artifacts are intentionally not committed. Large artifacts such as `.mrc`, `.mrcs`, `.pkl`, model weights, and downloaded datasets remain in `~/CryoAgent/real_runs` and are referenced through `artifact_manifest.json`.

## Included cases

### empiar10017_first1

One-micrograph same-dataset smoke case.

Chain:

```text
EMPIAR-10017 micrograph
→ Topaz particle coordinates
→ RELION particle extraction
→ RELION reconstruction
→ map2pdb dry-run routing
→ rule-based decision trace
→ LLM-supervised audit
