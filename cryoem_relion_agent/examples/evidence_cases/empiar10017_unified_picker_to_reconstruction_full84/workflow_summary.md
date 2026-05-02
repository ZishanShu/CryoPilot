# EMPIAR-10017 Closed-Loop Picker → Reconstruction Workflow

## Scientific caveat

This pipeline implements **picker-stage backend selection and downstream
handoff**.  It does NOT claim final structure determination.  The
reconstruction map produced here is a technical handoff artefact for
downstream QC, not a validated cryo-EM structure.

## Stage status

| stage | name | status |
|---|---|---|
| 01 | Record Topaz failure evidence | ok |
| 02 | Picker fallback selection | ok |
| 03 | RELION LoG rerun + F1 | ok |
| 04 | Materialise extraction project | ok |
| 05 | relion_preprocess extraction | ok |
| 06 | relion_reconstruct reconstruction | ok |

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
- Reports: `reports/`
