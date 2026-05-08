# EMPIAR-10017 Downstream Gates Reuse Full84 GPU Plan Evidence

This fixture stores lightweight evidence for CryoAgent's downstream-gated workflow using reused upstream full84 results.

It reuses:
- RELION LoG full84 F1
- extracted particle count
- reconstruction map path

It adds:
- Class2D GPU-aware plan
- Refine3D GPU-aware plan
- Postprocess/FSC map quality gate
- map2pdb routing gate
- model validation gate
- final readiness report

Core result:
- upstream_reused = true
- Topaz t-4 F1 = 0.031848
- RELION LoG F1 = 0.649950
- extracted particles = 84879
- reconstruction map exists = true
- Class2D plan includes --gpu 0
- Refine3D plan includes --gpu 0
- final_readiness = MAP_QC_REQUIRED

No raw MRC, MRCS particle stacks, or reconstructed maps are included.
