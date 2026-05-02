# Stage 08 – 3D refinement

| key | value |
|---|---|
| status | planned |
| decision | PLAN_REFINE3D |
| mode | plan-only |
| map_path | /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/04_reconstruction/reconstruct_from_relion_log.mrc |
| particles_star | /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/03_extract_project/relion_project/Extract/job002_log/particles.star |
| n_particles | 84879 |
| map_artifact_from_reconstruct | True |
| refined_map_available | False |

## Command

```
/home/user/conda_envs/relion/bin/relion_refine --i /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/03_extract_project/relion_project/Extract/job002_log/particles.star --ref /home/user/CryoAgent/real_runs/empiar10017_unified_downstream_gates_full84_real_v3/04_reconstruction/reconstruct_from_relion_log.mrc --o Refine3D/job001/run --auto_refine --split_random_halves --particle_diameter 120.0 --ctf --ini_high 30 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --norm --scale
```
