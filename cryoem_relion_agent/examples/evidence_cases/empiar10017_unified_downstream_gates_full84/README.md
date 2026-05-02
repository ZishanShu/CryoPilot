# EMPIAR-10017 Unified Downstream-Gated Full84 Evidence

This fixture stores lightweight evidence for CryoAgent's unified closed-loop workflow:

Topaz QC failure
→ fallback picker selection
→ RELION LoG rerun
→ F1 evaluation
→ RELION extraction
→ RELION reconstruction
→ Class2D / Refine3D / Postprocess-FSC planning gates
→ map2pdb routing stop-gate
→ model validation gate
→ final readiness report

Core result:

Topaz t-4 full84 F1 = 0.031848
RELION LoG full84 F1 = 0.649950
RELION LoG predicted particles = 84879
RELION extracted particles = 84879
RELION reconstruction map produced = yes
Final readiness = MAP_QC_REQUIRED

This fixture does not include raw MRC files, MRCS particle stacks, or reconstructed maps.
