# LLM-supervised Result

{
  "accept_backend_execution": true,
  "accept_for_downstream": false,
  "risk_level": "high",
  "main_issue": "Runtime artifacts indicate picking/extraction/reconstruction completed, but scientific acceptance is not justified: only 10 micrographs and 1097 particles were used, the reconstruction is described as a smoke/test map, no FSC/resolution, CTF/motion QC, 2D/3D classification, particle-quality validation, or map interpretability evidence is reported, and map2pdb appears to be dry-run/recorded rather than scientifically executed.",
  "recommended_next_action": "Do not use this map for map-to-model or scientific conclusions. Audit the RELION logs for errors/warnings, visually inspect picks/extracted particles and the reconstruction, then rerun a proper cryo-EM workflow on a sufficiently large curated dataset with motion/CTF QC, 2D classification, 3D refinement/classification, gold-standard FSC validation, and only proceed to map2pdb if the map reaches acceptable resolution and interpretability."
}
