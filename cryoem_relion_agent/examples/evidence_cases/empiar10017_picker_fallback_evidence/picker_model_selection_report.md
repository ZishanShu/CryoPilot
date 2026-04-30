# Picker model selection report

## Trigger
- previous_backend: topaz_t-4
- selected_micro_f1: 0.031848
- qc_threshold_f1: 0.2
- picking_qc_passed: False

## Retrieved candidates
# Alternative picker candidates

| order | name | status | kind | notes |
|---|---|---|---|---|
| 1 | relion_log_autopick | available | rule_based_log_picker | RELION LoG/autopick available; can produce real coordinates and be evaluated with the existing F1 evaluator. |
| 2 | cryotransformer_picker | partial | deep_learning_picker | Local CryoTransformer repo and predict.py present, but inference command is not verified.  Treat as plan-only until a successful smoke run is recorded. |
| 3 | cryolo_picker | unavailable | deep_learning_picker | cryolo_predict.py missing; crYOLO is unavailable.  This is non-fatal: the stage will skip crYOLO and continue. |
| 4 | topaz_supervised_retrain | available | deep_learning_picker_retrain | topaz binary and GT coordinates available for supervised retraining. |
| 5 | template_picker | unavailable | template_matching_picker | Template picker script missing. |

## Selection
- selected: relion_log_autopick
- final_action: RUN_ALTERNATIVE_PICKER
- reason: Selected relion_log_autopick (available) per priority order.

## Final decision
- final_action: STOP_PLAN_ONLY_NO_VIABLE_PICKER
- reason: Best alternative F1=0.0 is below qc_threshold=0.2; scientific STOP preserved.
- relion_log_f1: 0.0

## RELION LoG autopick run
- mode: dry_run
- f1: 0.0
- predicted_count: 0
- gt_count: 24
- returncode: None
- error: None

## Notes
- Rule-based baseline behavior is unchanged by this stage.
- crYOLO unavailability is non-fatal.
- CryoTransformer is plan-only unless an inference smoke run is verified.
- Scientific STOP is preserved when best F1 < qc_threshold.
- This stage does not claim full autonomous structure determination.
