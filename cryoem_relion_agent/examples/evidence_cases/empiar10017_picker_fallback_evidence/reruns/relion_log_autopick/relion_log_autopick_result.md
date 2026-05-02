# RELION LoG autopick result

- mode: dry_run
- f1: 0.0
- precision: 0.0
- recall: 0.0
- predicted_count: 0
- gt_count: 24
- matched_count: 0
- match_radius_px: 80.0
- returncode: None
- runtime_sec: 0.0
- n_micrographs: 3
- evaluation_available: True
- error: None

## Command
```
examples/evidence_cases/empiar10017_picker_fallback_evidence/_simulated_backends/bin/relion_autopick --i examples/evidence_cases/empiar10017_picker_fallback_evidence/reruns/relion_log_autopick/micrographs.star --odir examples/evidence_cases/empiar10017_picker_fallback_evidence/reruns/relion_log_autopick/AutoPick/job001 --pickname autopick --LoG --LoG_diam_min 90.0 --LoG_diam_max 150.0 --lowpass 20 --LoG_adjust_threshold 0 --pipeline_control examples/evidence_cases/empiar10017_picker_fallback_evidence/reruns/relion_log_autopick/AutoPick/job001
```

Note: This stage does not claim final structure determination.
