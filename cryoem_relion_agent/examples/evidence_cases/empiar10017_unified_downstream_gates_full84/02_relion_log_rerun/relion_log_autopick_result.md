# RELION LoG autopick result

- mode: real_run
- f1: 0.64995
- precision: 0.481427
- recall: 1.0
- predicted_count: 84879
- gt_count: 40863
- matched_count: 40863
- match_radius_px: 80.0
- returncode: 0
- runtime_sec: 1366.637
- n_micrographs: 84
- evaluation_available: True
- error: None

## Command
```
/home/user/conda_envs/relion/bin/relion_autopick --i /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/02_relion_log_rerun/micrographs.star --odir /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/02_relion_log_rerun/AutoPick/job001 --pickname autopick --LoG --LoG_diam_min 90.0 --LoG_diam_max 150.0 --lowpass 20 --LoG_adjust_threshold 0 --pipeline_control /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/02_relion_log_rerun/AutoPick/job001
```

Note: This stage does not claim final structure determination.
