# Rule-based Same-Dataset Agent Workflow Result

## Status

- Dataset: EMPIAR-10017
- Micrographs: 1
- Topaz coordinates used: 115
- Extracted particles: 115
- Particle stack count: 1
- Reconstruction map exists: True
- Map2PDB mode: dry-run

## Artifacts

- RELION project: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full_first1_fixed_llm/relion_project`
- Micrographs STAR: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full_first1_fixed_llm/relion_project/micrographs.star`
- Particles STAR: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full_first1_fixed_llm/relion_project/Extract/job001_box/particles.star`
- Reconstruction map: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full_first1_fixed_llm/reconstruct/reconstruct_from_topaz_extract_box_smoke.mrc`
- Rule decisions JSONL: `/home/user/CryoAgent/real_runs/empiar10017_same_dataset_agent_full_first1_fixed_llm/rule_based/stage_decisions.jsonl`

## Interpretation

This run records a same-dataset agent workflow from Topaz particle picking coordinates to RELION particle extraction and RELION reconstruction. Map2PDB is invoked according to the selected mode.

This is a technical workflow smoke/batch case. Scientific quality still requires downstream QC.
