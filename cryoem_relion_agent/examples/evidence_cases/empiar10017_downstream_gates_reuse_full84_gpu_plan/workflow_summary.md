# EMPIAR-10017 Closed-Loop Picker → Reconstruction Workflow

## Scientific caveat

This pipeline implements **autonomous picker-to-reconstruction workflow**
with **closed-loop picker fallback and downstream handoff**, towards
autonomous cryo-EM structure determination.  It does NOT claim final
structure determination.  The reconstruction map produced here is a
technical handoff artefact; map-to-model routing is stop-gated when map
QC evidence (FSC / postprocess) is missing.

## Stage status

| stage | name | status |
|---|---|---|
| 01 | Record Topaz failure evidence | reused |
| 02 | Picker fallback selection | reused |
| 03 | RELION LoG rerun + F1 | reused |
| 04 | Materialise extraction project | reused |
| 05 | relion_preprocess extraction | reused |
| 06 | relion_reconstruct reconstruction | reused |
| 07 | Class2D plan / gate | planned |
| 08 | Refine3D plan / gate | planned |
| 09 | Postprocess / FSC gate | stop_gated |
| 10 | Map-to-PDB routing | ok |
| 11 | Model validation | skipped |

## Final readiness

**MAP_QC_REQUIRED**

## Key metrics

| metric | value |
|---|---|
| Topaz F1 (failed) | 0.031848 |
| RELION LoG F1 | 0.64995 |
| QC threshold F1 | 0.2 |
| QC passed | True |
| Predicted particles | 84879 |
| GT particles | 40863 |
| Matched particles | 40863 |
| Extracted particles | 84879 |
| Reconstruction map | True |

## Output locations

- Picker selection: `01_picker_selection/`
- RELION LoG rerun: `02_relion_log_rerun/`
- Extraction project: `03_extract_project/`
- Reconstruction: `04_reconstruction/`
- Class2D plan: `05_class2d/`
- Refine3D plan: `06_refine3d/`
- Postprocess / FSC: `07_postprocess/`
- Map-to-PDB routing: `08_map2pdb_routing/`
- Model validation: `09_model_validation/`
- Reports: `reports/`
