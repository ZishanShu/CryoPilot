# Alternative picker candidates

| order | name | status | kind | notes |
|---|---|---|---|---|
| 1 | relion_log_autopick | available | rule_based_log_picker | RELION LoG/autopick available; can produce real coordinates and be evaluated with the existing F1 evaluator. |
| 2 | cryotransformer_picker | unavailable | deep_learning_picker | CryoTransformer repo or predict.py missing. |
| 3 | cryolo_picker | unavailable | deep_learning_picker | cryolo_predict.py missing; crYOLO is unavailable.  This is non-fatal: the stage will skip crYOLO and continue. |
| 4 | topaz_supervised_retrain | unavailable | deep_learning_picker_retrain | topaz binary or GT coordinate paths missing. |
| 5 | template_picker | unavailable | template_matching_picker | Template picker script missing. |
