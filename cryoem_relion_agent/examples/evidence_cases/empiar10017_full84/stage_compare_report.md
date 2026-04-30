# Same-Dataset Agent Stage Comparison Report

## Case summary

- Dataset: EMPIAR-10017
- Workflow: same_dataset_particle_picking_to_map2pdb_agent_case
- Run root: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84`
- Micrographs: 84
- Topaz coordinates used: 7404
- Extracted particles: 7404
- Particle stack count: 84
- Reconstruction map exists: True
- Reconstruction map size bytes: 67109888

## Stage-by-stage comparison

| Stage | Backend | Rule-based status | Primary metric | LLM accepts execution | LLM accepts downstream | LLM risk |
| --- | --- | --- | --- | --- | --- | --- |
| particle_picking | topaz | accepted_for_extraction | topaz_coords_used=7404 | True | False | high |
| particle_extraction | relion_preprocess | accepted_for_reconstruction | extracted_particles=7404; particle_stacks=84 | True | False | high |
| reconstruction | relion_reconstruct | accepted_for_map2pdb | map_exists=True; map_size_bytes=67109888 | True | False | high |
| map2pdb |  | recorded | dry_run_recorded=true; cif_or_pdb_generated=false | False | False | high |

## Detailed stage notes

### particle_picking

- Real backend: `topaz`
- Rule-based status: `accepted_for_extraction`
- Rule-based reason: Topaz coordinate files exist and contain candidate particles; pass to RELION extraction smoke.
- Primary metric: `topaz_coords_used=7404`
- Main artifact: `Topaz .coord / converted .box coordinates`
- LLM accepts backend execution: `True`
- LLM accepts downstream scientific use: `False`
- LLM risk level: `high`
- LLM main issue: Topaz coordinate files were produced and used for 84 micrographs with 7404 total candidate particles, but the picks have not been scientifically validated. Counts vary widely across micrographs, several micrographs have very few picks, and there is no reported visual inspection, precision/recall assessment, CTF-aware screening, particle-quality assessment, or downstream 2D-class evidence that the picked coordinates correspond to true particles.
- LLM recommended next action: Accept the Topaz run as a completed backend picking step only. Before using the picks for scientific reconstruction, visually inspect representative micrographs and coordinates, verify pixel/coordinate conventions, remove poor micrographs, and run 2D classification after extraction to assess particle quality.

### particle_extraction

- Real backend: `relion_preprocess`
- Rule-based status: `accepted_for_reconstruction`
- Rule-based reason: Extraction accepted if particles.star contains particle image rows and particle stack files exist.
- Primary metric: `extracted_particles=7404; particle_stacks=84`
- Main artifact: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/relion_project/Extract/job001_box/particles.star`
- LLM accepts backend execution: `True`
- LLM accepts downstream scientific use: `False`
- LLM risk level: `high`
- LLM main issue: RELION extraction appears to have completed successfully with particles.star and 84 particle stacks containing 7404 particles, but extraction quality is not validated. There is no evidence of correct box size, centering, normalization, CTF metadata consistency, particle-stack integrity beyond file existence, or rejection of junk particles.
- LLM recommended next action: Accept extraction as technically executed, but do not accept these particles for scientific reconstruction yet. Verify extraction parameters and metadata, inspect particle images, confirm box size and coordinate scaling, then perform 2D classification and remove junk or damaged particles before reconstruction.

### reconstruction

- Real backend: `relion_reconstruct`
- Rule-based status: `accepted_for_map2pdb`
- Rule-based reason: Reconstruction accepted if RELION writes a non-empty MRC map.
- Primary metric: `map_exists=True; map_size_bytes=67109888`
- Main artifact: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/reconstruct/reconstruct_from_topaz_extract_box_smoke.mrc`
- LLM accepts backend execution: `True`
- LLM accepts downstream scientific use: `False`
- LLM risk level: `high`
- LLM main issue: RELION wrote a non-empty MRC reconstruction map, but it is explicitly a smoke/test reconstruction from only 84 micrographs and 7404 particles. There is no reported CTF validation, angular assignment/refinement, independent half-map reconstruction, gold-standard FSC, resolution estimate, local resolution, map sharpening, class/refinement diagnostics, or evidence of interpretable biological signal.
- LLM recommended next action: Treat the map as a technical smoke-test artifact only. Run proper particle cleaning, initial model or ab-initio reconstruction, 3D classification/refinement, independent half-map refinement, FSC validation, sharpening, and map-quality inspection before using any map for modeling.

### map2pdb

- Real backend: ``
- Rule-based status: `recorded`
- Rule-based reason: Map2PDB is invoked through CryoAgent run-next if requested; dry-run is allowed for audit trail.
- Primary metric: `dry_run_recorded=true; cif_or_pdb_generated=false`
- Main artifact: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/relion_project/.cryoem_agent/action_history.jsonl`
- LLM accepts backend execution: `False`
- LLM accepts downstream scientific use: `False`
- LLM risk level: `high`
- LLM main issue: The map2pdb stage was recorded as a dry-run/action-history event rather than an actual validated model-building execution. The reported result has return_code 0 but has_error_pattern is true, and no generated or validated atomic model is reported. The input map itself is not scientifically validated for model building.
- LLM recommended next action: Do not interpret this as a successful map-to-model result. First obtain a validated, interpretable reconstruction with resolution and FSC evidence. Then run a real map2pdb/model-building workflow, inspect logs for errors, and validate the resulting model against the map using standard model-quality and map-fit metrics.

## Interpretation

This report is data-oriented. It separates real backend execution, rule-based workflow progression, and LLM-supervised scientific auditing. A stage can be technically successful while still being rejected for scientific downstream use by the LLM supervisor.

If Map2PDB was run in dry-run mode, this report records routing and audit evidence only. A real map-to-model result requires `--map2pdb-mode real` and a generated CIF/PDB artifact.

- CSV table: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/stage_compare_table.csv`