# LLM Stage-wise Supervised Audit

| Stage | Accept execution | Accept downstream | Risk | Main issue |
| --- | --- | --- | --- | --- |
| particle_picking | True | False | high | Topaz coordinate files were produced and used for 84 micrographs with 7404 total candidate particles, but the picks have not been scientifically validated. Counts vary widely across micrographs, several micrographs have very few picks, and there is no reported visual inspection, precision/recall assessment, CTF-aware screening, particle-quality assessment, or downstream 2D-class evidence that the picked coordinates correspond to true particles. |
| particle_extraction | True | False | high | RELION extraction appears to have completed successfully with particles.star and 84 particle stacks containing 7404 particles, but extraction quality is not validated. There is no evidence of correct box size, centering, normalization, CTF metadata consistency, particle-stack integrity beyond file existence, or rejection of junk particles. |
| reconstruction | True | False | high | RELION wrote a non-empty MRC reconstruction map, but it is explicitly a smoke/test reconstruction from only 84 micrographs and 7404 particles. There is no reported CTF validation, angular assignment/refinement, independent half-map reconstruction, gold-standard FSC, resolution estimate, local resolution, map sharpening, class/refinement diagnostics, or evidence of interpretable biological signal. |
| map2pdb | False | False | high | The map2pdb stage was recorded as a dry-run/action-history event rather than an actual validated model-building execution. The reported result has return_code 0 but has_error_pattern is true, and no generated or validated atomic model is reported. The input map itself is not scientifically validated for model building. |

## Overall assessment

```json
{
  "accept_backend_execution": true,
  "accept_for_scientific_claim": false,
  "risk_level": "high",
  "summary": "The picking, extraction, and reconstruction backend commands appear to have produced expected technical artifacts, but the workflow is only a smoke-test style same-dataset run and is not scientifically validated. The reconstruction is based on a small subset of 84 micrographs and 7404 particles with no reported particle cleanup, 2D/3D classification, refinement, half-map FSC, resolution, sharpening, or map interpretability assessment. The map2pdb step was only a dry-run and should not be considered model building. These outputs are acceptable as backend execution evidence, except for actual map2pdb execution, but they must not be used for downstream scientific claims or structural interpretation."
}
```