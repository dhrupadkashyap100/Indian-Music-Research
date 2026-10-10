# TRF Validation & Optimization Report

## Best Method Selected (Validation)
- **Transform**: PCA128_Whiten
- **Raag Aggregation**: max
- **Temporal Aggregation**: mean

## TRF Test Performance (Sealed Set)
- **Top-1 Accuracy**: 0.1532
- **Macro F1**: 0.1012
- **Mean Top-1/Top-2 Margin**: 0.064614

## Margin Analysis
- Correct Predictions Mean Margin: 0.183454
- Incorrect Predictions Mean Margin: 0.043115
*Conclusion: A statistically higher margin correlates strongly with correct retrieval.*

## SARAGA External Validation
- **SARAGA Top-1 Accuracy**: 0.0741
- **SARAGA Macro F1**: 0.0538

## Limitations
Due to the strict recording-level split, some rare classes did not have representation in the Validation/Test set. SARAGA transfer indicates the presence of a domain gap, typical for zero-shot candidate retrieval.
