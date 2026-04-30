# Alternative picker candidates

| order | name | status | kind | notes |
|---|---|---|---|---|
| 1 | relion_log_autopick | available | rule_based_log_picker | RELION LoG/autopick available; can produce real coordinates and be evaluated with the existing F1 evaluator. |
| 2 | cryotransformer_picker | partial | deep_learning_picker | Local CryoTransformer repo and predict.py present, but inference command is not verified.  Treat as plan-only until a successful smoke run is recorded. |
| 3 | cryolo_picker | unavailable | deep_learning_picker | cryolo_predict.py missing; crYOLO is unavailable.  This is non-fatal: the stage will skip crYOLO and continue. |
| 4 | topaz_supervised_retrain | available | deep_learning_picker_retrain | topaz binary and GT coordinates available for supervised retraining. |
| 5 | template_picker | unavailable | template_matching_picker | Template picker script missing. |
