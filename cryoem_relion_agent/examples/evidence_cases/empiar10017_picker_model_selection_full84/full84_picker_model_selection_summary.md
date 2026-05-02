# EMPIAR-10017 Full84 Picker Model-Selection Result

## Purpose

This experiment validates CryoAgent's LLM-supervised closed-loop fallback mechanism at the particle-picking stage.

The goal is not final structure determination. The goal is to test whether CryoAgent can detect a failed picker configuration, retrieve an alternative backend, execute the fallback picker, evaluate the result with GT-based metrics, and select the better backend.

## Trigger: Topaz threshold sweep failed QC

The previous Topaz threshold sweep selected `topaz_t-4` by micro-F1, but the result remained below the QC threshold.

| Backend/config | Subset | F1 | QC threshold | Decision |
| --- | ---: | ---: | ---: | --- |
| Topaz t-4 | full84 | 0.031848 | 0.2 | failed QC |

This triggered:

```text
FALLBACK_TO_ALTERNATIVE_PICKER_OR_RETRAIN
