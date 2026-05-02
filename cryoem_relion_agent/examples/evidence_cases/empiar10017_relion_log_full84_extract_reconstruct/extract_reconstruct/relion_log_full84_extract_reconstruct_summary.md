# EMPIAR-10017 RELION LoG Full84 Extraction and Reconstruction Result

## Purpose

This experiment validates the downstream handoff after CryoAgent selected RELION LoG/autopick as the fallback picker.

## Workflow chain

Topaz threshold sweep failed QC
→ CryoAgent selected RELION LoG/autopick
→ RELION LoG full84 picking
→ RELION particle extraction
→ RELION reconstruction

## Input picker result

| Metric | Value |
| --- | ---: |
| Picker | RELION LoG/autopick |
| Micrographs | 84 |
| Predicted particles | 84879 |
| GT particles | 40863 |
| Picking F1 | 0.643890 |

## Extraction result

| Metric | Value |
| --- | ---: |
| Extracted particle rows | 84879 |
| Particle STAR | /home/user/CryoAgent/real_runs/empiar10017_relion_log_full84_extract_reconstruct/relion_project/Extract/job002_log_full84/particles.star |

## Reconstruction result

| Metric | Value |
| --- | --- |
| Reconstruction map exists | yes |
| Map size | 65M |
| Map path | /home/user/CryoAgent/real_runs/empiar10017_relion_log_full84_extract_reconstruct/reconstruct_full84/reconstruct_from_relion_log_full84.mrc |

## Interpretation

This result shows that the fallback picker selected by CryoAgent can support downstream RELION extraction and reconstruction on the EMPIAR-10017 full84 subset.

This is a workflow-level handoff and backend-selection result. It should not be described as a final scientific cryo-EM structure.
