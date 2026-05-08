# Stage 07 – Class2D

| key | value |
|---|---|
| status | planned |
| decision | PLAN_CLASS2D |
| mode | plan-only |
| K | 50 |
| particles_star | /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/03_extract_project/relion_project/Extract/job002_log/particles.star |
| n_particles | 84879 |
| output_dir | Class2D/job001/run |
| gpu_requested | True |
| gpu_supported | True |
| gpu_devices | 0 |

## Command

```
/home/user/conda_envs/relion/bin/relion_refine --i /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/03_extract_project/relion_project/Extract/job002_log/particles.star --o Class2D/job001/run --iter 25 --K 50 --particle_diameter 120.0 --ctf --pool 10 --norm --scale --gpu 0
```
