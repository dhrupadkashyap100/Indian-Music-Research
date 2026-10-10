# Figures and Tables Specifications

## Figures

**Figure 1: Complete System Architecture**
- **Purpose:** Illustrate the 1884-dimensional feature extraction and parallel decision branches.
- **Source:** Defined in evidence master; extract from `diagrams_and_charts.md`.
- **Caption:** Flow diagram of the Hybrid Geometric-Discriminative architecture, illustrating the $0.75 / 0.25$ arithmetic fusion.

**Figure 2: Dataset Isolation Protocol**
- **Purpose:** Prove strict leakage prevention.
- **Source:** Defined in evidence master; extract from `diagrams_and_charts.md`.
- **Caption:** Data flowchart demonstrating the strict isolation of the sealed TRF Test set and Folk recordings during optimization.

**Figure 3: Feature Space Sparsity**
- **Purpose:** Visually contrast the dense Transition Matrix against the highly sparse N=3 Trigrams.
- **Axes:** Feature Index vs Magnitude.
- **Caption:** Heatmap of the 1884-dimensional extraction, emphasizing the sparsity of third-order melodic sequences.

**Figure 4: Model Comparison**
- **Purpose:** Compare Control, Neural, and Hybrid performance on Validation and SARAGA.
- **Axes:** X = Models, Y = Accuracy (%).
- **Caption:** Performance comparison illustrating the collapse of Metric Learning relative to the Geometric and Hybrid models.

**Figure 5: Performance Trajectory**
- **Purpose:** Display classical evaluation accuracy.
- **Axes:** X = Evaluation Split (Val, SARAGA, Test), Y = Accuracy (%).
- **Caption:** Accuracy trajectory of the final Hybrid SVM B model across splits.

**Figure 6: Folk Confidence Distribution**
- **Purpose:** Demonstrate structural domain shift.
- **Source:** `FINAL_CHART_CONFIDENCE_DISTRIBUTION.csv`
- **Caption:** Pie chart showing 86.5% of Folk recordings categorizing as structurally ambiguous relative to the classical boundaries.

**Figure 7: Top-5 Folk Structural-Affinity Hypotheses**
- **Purpose:** Show the most frequent computational mappings.
- **Source:** `FINAL_CHART_RAAG_DISTRIBUTION.csv`
- **Axes:** X = Count, Y = Raag Hypothesis.
- **Caption:** Distribution of the top five generated classical structural hypotheses for the Folk corpus.

**Figure 8: Genre $\times$ Raag Affinity Heatmap**
- **Purpose:** Visualize regional mappings (e.g., Baul $\rightarrow$ Malkauns).
- **Source:** `FINAL_CHART_GENRE_RAAG.csv`
- **Caption:** Heatmap illustrating the density of specific predicted Raag hypotheses grouped by Folk genre.

---

## Tables

**Table 1: Dataset Inventory**
| Dataset | Recordings | Classes | Purpose | Ground Truth |
| :--- | :--- | :--- | :--- | :--- |
| TRF Train | 635 | 61 | Training/reference | Yes |
| TRF Validation | 223 | 61 | Model selection | Yes |
| TRF Test | 245 | 61 | Final evaluation | Yes |
| SARAGA | 54 | 61-class mapped | External generalization | Yes |
| Folk | 328 | N/A | Structural-affinity inference | No |

**Table 2: 1884-Dimensional Feature Representation**
| Feature | Dimensions | Semantic Interpretation | Weight ($\alpha$) |
| :--- | :--- | :--- | :--- |
| Transition Matrix | 144 | Melodic Grammar (Chalan) | 0.75 |
| Pitch-Class Dist | 12 | Pitch Hierarchy (Vadi/Samvadi) | 0.20 |
| N=3 Trigrams | 1728 | Localized Sequential Identity | 0.05 |

**Table 3: Model Comparison**
| Model | TRF Validation | SARAGA Generalization | Status |
| :--- | :--- | :--- | :--- |
| Geometric Control | 34.93% | 52.83% | Baseline |
| Linear Metric Learning | 33.49% | 45.28% | Rejected (Overfit) |
| Hybrid SVM B | 39.23% | 56.60% | Selected |

**Table 4: Negative Optimization Ablations**
| Experiment | Observed Result | Interpretation / Failure Mode |
| :--- | :--- | :--- |
| PCA | Validation collapsed | Destroyed sparse trigram sequence geometry |
| LDA | SARAGA collapsed | Overfit severe class sparsity ($N \approx 10$) |
| Hard Negative Mining | Validation plunged < 20% | Gradient collapse; distorted loss manifold |
| Mean / Medoid Prototypes | Validation dropped < 31% | Centroid averaging destroys intra-class phrase diversity |

**Table 5: Final Evaluation Performance**
| Metric | Value | Model |
| :--- | :--- | :--- |
| TRF Validation | 39.23% | Hybrid SVM B |
| SARAGA Generalization | 56.60% | Hybrid SVM B |
| TRF Test Accuracy | 45.02% | Hybrid SVM B |

**Table 6: Selective Classification Thresholds**
| Hybrid Margin ($\Delta_{1,2}$) | CV Accuracy (TRF Train) | Folk Inference Tier |
| :--- | :--- | :--- |
| $> 0.12$ | 100% | HIGH |
| $0.05 - 0.12$ | 98.8% | MODERATE |
| $\leq 0.05$ | N/A | AMBIGUOUS |

**Table 7: Folk Confidence Distribution**
| Tier | Recordings | Percentage |
| :--- | :--- | :--- |
| HIGH | 6 | 1.8% |
| MODERATE | 38 | 11.6% |
| AMBIGUOUS | 283 | 86.5% |

**Table 8: Top Folk Structural-Affinity Hypotheses**
| Hypothesis | Frequency Count | Percentage of Corpus |
| :--- | :--- | :--- |
| Malkauns | 30 | 9.2% |
| Pahadi | 22 | 6.7% |
| Megh Malhar | 21 | 6.4% |
| Durga | 20 | 6.1% |
| Suddha Kalyan | 16 | 4.9% |

**Table 9: Ambiguous Topological Proximities**
| Primary Hypothesis | Secondary Hypothesis | Vacillation Frequency |
| :--- | :--- | :--- |
| Pahadi | Suddha Kalyan | 14 |
| Megh | Megh Malhar | 7 |
| Durga | Pahadi | 6 |
