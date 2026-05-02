# Picker model selection comparison

| model | kind | status | best_f1 | qc_threshold_f1 | selected |
|---|---|---|---|---|---|
| topaz_t-4 | topaz_threshold_sweep | failed_qc | 0.031848 | 0.2 | False |
| relion_log_autopick | rule_based_log_picker | available | 0.64389 | 0.2 | True |
| cryotransformer_picker | deep_learning_picker | partial | None | 0.2 | False |
| cryolo_picker | deep_learning_picker | unavailable | None | 0.2 | False |
| topaz_supervised_retrain | deep_learning_picker_retrain | available | None | 0.2 | False |
| template_picker | template_matching_picker | unavailable | None | 0.2 | False |
