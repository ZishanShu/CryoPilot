# QC-aware Final Selection

- Picking F1 summary: `/home/user/CryoAgent/real_runs/empiar10017_topaz_threshold_sweep_f1_eval/topaz_threshold_sweep_f1_summary.csv`
- Selection metric: `picking_micro_f1`
- Selected backend execution case: `topaz_t-4`
- Selected threshold: `-4`
- Selected micro-F1: `0.031848`
- QC threshold F1: `0.2`
- Picking QC passed: `False`
- Final action: `FALLBACK_TO_ALTERNATIVE_PICKER_OR_RETRAIN`

## Reason

Selected topaz_t-4 as the best Topaz threshold-sweep configuration by micro-F1=0.031848, but this remains below QC threshold 0.200000. Preserve scientific STOP gate and retrieve another particle-picking fallback.

## QC-aware comparison table

| case_id | config | threshold | n_micrographs | n_extracted_particles | TP | FP | FN | precision | recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empiar10017_same_dataset_full84 | empiar10017_same_dataset_full84 | -6 | 84 | 7404 | 787 | 7776 | 40076 | 0.091907 | 0.019259 | 0.031846 |
| topaz_t-8 | threshold=-8 | -8 | 84 | 7409 | 787 | 7781 | 40076 | 0.091853 | 0.019259 | 0.031842 |
| topaz_t-6 | threshold=-6 | -6 | 84 | 7404 | 787 | 7776 | 40076 | 0.091907 | 0.019259 | 0.031846 |
| topaz_t-4 | threshold=-4 | -4 | 84 | 7402 | 787 | 7773 | 40076 | 0.091939 | 0.019259 | 0.031848 |
| topaz_t-2 | threshold=-2 | -2 | 84 | 7397 | 786 | 7768 | 40077 | 0.091887 | 0.019235 | 0.031811 |
| topaz_t0 | threshold=0 | 0 | 84 | 7393 | 786 | 7763 | 40077 | 0.091941 | 0.019235 | 0.031814 |

## Interpretation

When ground-truth picking metrics are available, CryoAgent should select by micro-F1 rather than extracted-particle count. If the best F1 remains below the QC threshold, the system must preserve the scientific STOP gate and retrieve another fallback picker.