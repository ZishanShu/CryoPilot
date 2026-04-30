# LLM-supervised Result

{
  "accept_backend_execution": true,
  "accept_for_downstream": false,
  "risk_level": "high",
  "main_issue": "Runtime artifacts indicate picking, extraction, and a reconstruction command completed, but the result is not scientifically validated for map2pdb/model building. The reconstruction appears to be a smoke/test map from only 84 micrographs and 7,404 particles, with no reported CTF validation, particle-quality assessment, 2D classification, ab-initio/3D refinement, gold-standard FSC, local resolution, map sharpening, or evidence that the map contains interpretable biological signal. The map2pdb step is recorded as a dry-run/action history rather than a validated modeling result.",
  "recommended_next_action": "Do not use this map for downstream model building yet. Inspect picks and extracted particles, run 2D classification and remove junk, perform proper 3D refinement/reconstruction with independent half-maps and FSC validation, report resolution and map quality metrics, then only proceed to map2pdb if the refined map is interpretable and passes validation."
}
