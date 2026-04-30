# Rule-based Same-Dataset Agent Workflow Result

## Status

- Dataset: EMPIAR-10017
- Micrographs: 84
- Topaz coordinates used: 7404
- Extracted particles: 7404
- Particle stack count: 84
- Reconstruction map exists: True
- Map2PDB mode: dry-run

## Artifacts

- RELION project: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/relion_project`
- Micrographs STAR: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/relion_project/micrographs.star`
- Particles STAR: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/relion_project/Extract/job001_box/particles.star`
- Reconstruction map: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/reconstruct/reconstruct_from_topaz_extract_box_smoke.mrc`
- Rule decisions JSONL: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full84/rule_based/stage_decisions.jsonl`

## Interpretation

This run records a same-dataset agent workflow from Topaz particle picking coordinates to RELION particle extraction and RELION reconstruction. Map2PDB is invoked according to the selected mode.

This is a technical workflow smoke/batch case. Scientific quality still requires downstream QC.
