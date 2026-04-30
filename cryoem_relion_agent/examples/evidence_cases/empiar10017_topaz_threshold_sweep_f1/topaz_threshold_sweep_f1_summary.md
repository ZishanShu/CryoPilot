# Topaz Threshold Sweep F1 Evaluation

- Pairs CSV: `/home/user/CryoAgent/real_runs/empiar10017_picking_full/tables/empiar10017_full_pairs.csv`
- Matching radius: 20.0 px
- Evaluated micrographs with GT: 84
- Missing/empty GT rows: 0
- Selected threshold by micro-F1: `-4`

## Summary table

| threshold | n_micrographs | total_pred | total_gt | TP | FP | FN | micro_precision | micro_recall | micro_F1 | macro_F1 | selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| -8 | 84 | 8568 | 40863 | 787 | 7781 | 40076 | 0.0919 | 0.0193 | 0.0318 | 0.0291 | False |
| -6 | 84 | 8563 | 40863 | 787 | 7776 | 40076 | 0.0919 | 0.0193 | 0.0318 | 0.0291 | False |
| -4 | 84 | 8560 | 40863 | 787 | 7773 | 40076 | 0.0919 | 0.0193 | 0.0318 | 0.0291 | True |
| -2 | 84 | 8554 | 40863 | 786 | 7768 | 40077 | 0.0919 | 0.0192 | 0.0318 | 0.0291 | False |
| 0 | 84 | 8549 | 40863 | 786 | 7763 | 40077 | 0.0919 | 0.0192 | 0.0318 | 0.0291 | False |

## Interpretation

This table evaluates particle-picking quality using ground-truth coordinate matching. The closed-loop selector should prefer micro-F1 when GT is available, rather than selecting only by extracted particle count.

## Output files

- Summary CSV: `/home/user/CryoAgent/real_runs/empiar10017_topaz_threshold_sweep_f1_eval/topaz_threshold_sweep_f1_summary.csv`
- Per-micrograph CSV: `/home/user/CryoAgent/real_runs/empiar10017_topaz_threshold_sweep_f1_eval/topaz_threshold_sweep_f1_per_micrograph.csv`
- JSON: `/home/user/CryoAgent/real_runs/empiar10017_topaz_threshold_sweep_f1_eval/topaz_threshold_sweep_f1_summary.json`