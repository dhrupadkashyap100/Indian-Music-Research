# A Hybrid Geometric–Discriminative Framework for Raag Classification and Structural Raag Affinity Analysis in Indian Folk Music

**Abstract**

Identifying the melodic Raag framework underlying Indian traditional music is a challenging Music Information Retrieval (MIR) problem because Raag identity is determined by complex melodic structures rather than solely by pitch-class content. The problem becomes particularly challenging in low-data settings, where conventional deep-learning approaches can overfit and fail to generalize across musical traditions.

This study presents a hybrid Raag classification framework combining domain-informed melodic similarity with supervised discriminative learning. The proposed representation integrates a 144-dimensional melodic transition representation, a 12-dimensional pitch-class distribution, and a 1,728-dimensional third-order melodic n-gram representation, producing an 1,884-dimensional feature space. An initial cosine-similarity retrieval system using query-wise normalization, weighted feature fusion, temporal aggregation, and maximum reference similarity established a strong non-parametric baseline. Several alternative approaches, including PCA, LDA, supervised metric learning, prototype-based retrieval, reference aggregation, score normalization, and tree-based classifiers, were systematically evaluated under strict train/validation/test isolation.

The final system combines the geometric retrieval score with an RBF-kernel Support Vector Machine using a 75:25 fusion between the geometric and discriminative components. The resulting Hybrid SVM achieved 39.23% Top-1 accuracy on the development validation set and 56.60% on the external SARAGA estimated-tonic evaluation. After satisfying the predefined development and cross-dataset generalization criteria, the frozen model was evaluated once on the sealed TRF test set, achieving 45.02% Top-1 accuracy.

The model was subsequently applied to 328 Indian Folk recordings to investigate structural Raag affinities. Of these, 327 recordings produced valid predictions, while one recording could not be processed. Importantly, the Folk predictions are treated as computational hypotheses rather than musicological ground truth. The system assigned an ambiguous/OOD confidence status to 86.5% of valid Folk recordings, demonstrating substantial domain shift between the classical reference corpus and Folk recordings. This work therefore contributes both a hybrid low-data Raag classification framework and a reproducible methodology for investigating classical Raag structural affinities in out-of-domain Folk music.

**Keywords:** Music Information Retrieval, Indian classical music, Raag classification, Hindustani music, melodic representation, cosine retrieval, Support Vector Machine, domain shift, Folk music, hybrid classification

---

## 1. Introduction

Indian classical music presents a distinctive challenge for automatic music analysis because its fundamental musical organization differs substantially from the assumptions underlying many conventional Western music classification systems. Rather than relying primarily on fixed scales or harmonic progressions, Hindustani music organizes melodic expression around Raag frameworks characterized by characteristic note usage, melodic transitions, phrase structures, pitch hierarchy, and context-dependent melodic behavior.

Automatic Raag identification is therefore not simply a pitch-class classification problem. Two Raags may contain substantially overlapping pitch collections while remaining distinguishable through their characteristic transitions and melodic movements. Conversely, performances of the same Raag may differ considerably because of performer interpretation, ornamentation, tonic selection, tempo, recording conditions, and improvisational variation.

These challenges become more severe when the available labeled training data are limited. In the present study, the primary classical training corpus contained 635 recordings distributed across 61 Raag classes, corresponding to approximately 10 recordings per class. Such a data regime creates a substantial risk of overfitting for high-capacity neural models.

The central objective of this work was therefore to determine whether a domain-informed, low-parameter representation could provide robust Raag discrimination under severe data constraints.

The study additionally investigates whether a classifier trained on classical music can provide meaningful structural hypotheses about Raag affinities in Indian Folk music. Rather than treating these predictions as ground-truth annotations, the Folk inference stage is explicitly framed as an analysis of structural similarity and model confidence under domain shift.

The main contributions of this work are:
1. An 1,884-dimensional melodic representation combining transition structure, pitch-class information, and third-order melodic n-grams.
2. A non-parametric geometric retrieval framework designed for low-data Raag recognition.
3. A systematic comparison of neural and classical learning approaches under strict dataset isolation.
4. A Hybrid Geometric–SVM classifier that achieved 45.02% Top-1 accuracy on the sealed TRF test set.
5. Cross-dataset evaluation using the SARAGA corpus, achieving 56.60% estimated-tonic Top-1 performance.
6. A controlled Folk inference framework that explicitly models ambiguity and out-of-distribution behavior rather than presenting computational predictions as musicological ground truth.
7. A publication-ready Folk dataset containing model-derived Raag hypotheses for 327 successfully processed recordings.

---

## 2. Related Work

### 2.1 Music Information Retrieval and Raag Recognition
Automatic analysis of Indian classical music has received increasing attention within Music Information Retrieval. Unlike genre classification, Raag recognition requires modeling melodic structure, pitch organization, and temporal relationships.

Previous systems have investigated pitch-class profiles, chromagrams, melodic contours, tonic-normalized pitch representations, sequential models, convolutional neural networks, recurrent neural networks, and transformer-based architectures.

However, many approaches depend on sufficiently large labeled datasets. This creates a fundamental challenge for Raag recognition when the number of recordings per class is small.

### 2.2 Pitch-Based Representations
Pitch-class representations provide compact descriptions of the pitch material present in a recording. They are useful because they provide some degree of transposition invariance after tonic normalization.

However, pitch-class distributions alone cannot fully capture Raag identity. Several Raags may share similar pitch collections while differing in characteristic melodic movements.

### 2.3 Sequential and Melodic Representations
Temporal melodic representations attempt to capture how notes evolve through time. Transition-based representations can encode the probability of movement between discretized pitch states, while n-gram representations capture recurring local melodic sequences.

These representations are particularly attractive for Raag analysis because melodic identity is strongly related to characteristic note transitions and phrase structures.

### 2.4 Deep Learning for Music Classification
CNN, CRNN, recurrent, and transformer-based architectures have demonstrated strong performance in many MIR tasks. Nevertheless, high-capacity models can be problematic in small-data regimes.

In this study, supervised metric-learning models were explicitly evaluated rather than assumed to be superior. Their performance was compared against a hand-engineered retrieval system under identical data-isolation constraints.

---

## 3. Dataset

### 3.1 Classical Raag Dataset
The principal development corpus consisted of 635 training recordings spanning 61 Raag classes. The small number of recordings per class created a highly constrained supervised learning environment.

The dataset was divided into training, validation, and test partitions. The test partition remained sealed throughout model development and was accessed only after a candidate satisfied the predefined validation and external-generalization criteria.

### 3.2 SARAGA Evaluation
SARAGA was used as an external evaluation corpus to measure cross-dataset generalization.

The external evaluation was particularly important because a model that performs well only on the development corpus may exploit recording-specific or corpus-specific characteristics rather than learning transferable melodic structure.

### 3.3 Indian Folk Corpus
The final inference corpus contained 328 Indian Folk recordings.

The Folk corpus was never used for model training, hyperparameter optimization, or model selection.

Of the 328 recordings:
- 327 were successfully processed.
- 1 recording could not be processed because sufficient voiced audio could not be extracted.

Therefore, all reported Folk predictions correspond to the 327 successfully processed recordings.

---

## 4. Feature Representation

The final representation consisted of three complementary components.

### 4.1 Melodic Transition Representation
A 144-dimensional transition representation was used to encode melodic movement between discretized pitch states.
Rather than considering only which notes occur, the representation captures how melodic states transition between one another.
This provides information that is unavailable from a simple pitch-class histogram.

### 4.2 Pitch-Class Distribution
A 12-dimensional pitch-class representation captures the relative occurrence of pitch classes.
Tonic-relative normalization was used so that the representation could be compared across recordings with different absolute tonic frequencies.

### 4.3 Third-Order Melodic N-Grams
A 1,728-dimensional N=3 representation was constructed from third-order melodic sequences.
The N=3 representation was intentionally retained at its original dimensionality because preliminary PCA experiments substantially degraded retrieval performance. This result suggested that the sparse local sequence information contained in the original representation was important for the task.

### 4.4 Combined Representation
The complete representation was:
[ 144 + 12 + 1728 = 1884 ] dimensions.

The final feature weighting was:
[ S_{feature} = 0.75S_{transition} + 0.20S_{pitch-class} + 0.05S_{N3} ]

This weighting was retained throughout the final frozen system.

---

## 5. Initial Geometric Retrieval Model

The initial classifier was deliberately designed without trainable parameters.
For each query recording, similarity was computed against the classical training reference bank.

The system used:
- weighted transition similarity,
- pitch-class similarity,
- N=3 similarity,
- query-wise score normalization,
- temporal aggregation,
- maximum reference similarity.

The resulting system achieved a development validation Top-1 accuracy of approximately 34.93% under the original control evaluation.
This baseline became the central reference point for subsequent experiments.

---

## 6. Systematic Optimization Experiments

A major objective of this study was to determine whether more sophisticated models could improve upon the hand-engineered retrieval geometry.

### 6.1 PCA
PCA dimensionality reduction was evaluated at multiple dimensionalities.
The experiments produced substantial performance degradation.
The result indicated that dimensionality reduction was inappropriate for the sparse N=3 representation because compression removed information relevant to melodic sequence discrimination.
Consequently, PCA was rejected.

### 6.2 LDA
Supervised Linear Discriminant Analysis was also evaluated.
Although supervised dimensionality reduction can theoretically improve class separation, the small number of recordings per class caused severe overfitting.
LDA therefore failed to provide useful cross-dataset generalization and was rejected.

### 6.3 Metric Learning
Lightweight neural metric-learning architectures were evaluated using triplet margin loss with multiple negative-mining strategies.
The best metric-learning candidate achieved approximately:
- 33.49% TRF validation Top-1
- 45.28% SARAGA Top-1

This was inferior to the geometric control.
Hard-negative mining was particularly unstable and caused substantial validation degradation.
These experiments demonstrated that adding trainable parameters did not automatically improve generalization in the low-data regime.

### 6.4 Prototype-Based Retrieval
Several non-neural retrieval geometries were subsequently evaluated, including:
- Mean prototypes
- Medoids
- K-means prototypes
- Top-K reference aggregation
- Reference-quality pruning
- Alternative feature weights
- Score normalization
- Temporal consistency penalties

None provided a sufficiently consistent improvement over the existing retrieval system.
The maximum-similarity reference strategy remained the strongest geometric configuration.

---

## 7. Classical Supervised Decision Models

The next experimental stage investigated whether a supervised classifier could learn decision boundaries complementary to the geometric retrieval system.

Tree-based methods such as ExtraTrees and gradient boosting demonstrated severe overfitting.
The RBF-kernel Support Vector Machine produced substantially better generalization.

The selected configuration was:
[ C=10 ] with: [ \gamma = \text{scale} ]

The SVM operated on the same 1,884-dimensional feature representation used by the geometric model.

---

## 8. Hybrid Geometric–SVM Model

The central contribution of the study is the combination of two complementary decision mechanisms.

The geometric component provides physically interpretable melodic similarity.
The SVM provides a supervised non-linear decision boundary.

The final hybrid score was:
[ S_{hybrid} = 0.75S_{geometric} + 0.25S_{SVM} ]
where the SVM probabilities were calibrated using training-only cross-validation.

This architecture combines a domain-informed retrieval mechanism with a discriminative classifier without replacing the original geometric model.

---

## 9. Experimental Results

The Hybrid SVM achieved:

| Evaluation | Top-1 Accuracy |
| :--- | :--- |
| TRF Validation | 39.23% |
| SARAGA Estimated-Tonic | 56.60% |
| Sealed TRF Test | 45.02% |

The final model therefore exceeded the historical 43.10% control benchmark on the sealed test evaluation.
The improvement was particularly important because the candidate was selected using development and cross-dataset criteria before the test set was accessed.
The sealed test set was evaluated only once after the final candidate passed the predefined development gates.

---

## 10. Error Complementarity

The success of the hybrid model suggests that the geometric and SVM components capture partially different aspects of Raag discrimination.
The geometric model provides a similarity-based neighborhood derived from melodic structure.
The SVM learns non-linear class boundaries in the same feature space.

Consequently, cases where the geometric model produces an incorrect ranking can sometimes be corrected by the SVM, while the geometric component provides a stabilizing influence when the SVM encounters ambiguous examples.
This complementarity provides a possible explanation for why the hybrid model outperformed either mechanism alone.

---

## 11. Folk Music Inference

After model development was completed, the frozen Hybrid SVM was applied to the Indian Folk corpus.
No Folk recordings were used during training, model selection, or hyperparameter tuning.

The objective was not to claim that the predicted Raag represents objective musical ground truth.
Instead, the system was used as a structural Raag-affinity hypothesis generator.

For each recording, the system produced:
- Top-1 predicted Raag
- Top-2 predicted Raag
- Top-3 predicted Raag
- Hybrid score
- Top-1/Top-2 margin
- Confidence tier
- Model agreement information

---

## 12. Confidence and Domain Shift

The Folk inference produced a striking confidence distribution.
Among 327 successfully processed recordings:

| Confidence tier | Recordings |
| :--- | :--- |
| High | 6 |
| Moderate | 38 |
| Ambiguous | 283 |

Thus, approximately 86.5% of valid Folk recordings were classified as ambiguous under the predefined confidence thresholds.

This observation should not be interpreted as proof that Folk music lacks Raag identity.
Instead, it provides evidence that the Folk corpus frequently lies outside the distribution represented by the classical training corpus.
The model therefore frequently lacked sufficient evidence to make a high-confidence classical Raag assignment.
This behavior is scientifically preferable to presenting forced predictions as certain annotations.

---

## 13. Folk Raag Affinity Distribution

Among the computational hypotheses, several Raags appeared frequently.
The most frequently predicted classes included:
- Malkauns
- Pahadi
- Megh Malhar
- Durga
- Suddha Kalyan

These results should be interpreted cautiously.
A high frequency of a predicted Raag does not demonstrate that the corresponding Folk recordings objectively belong to that Raag.
Instead, it indicates that the recordings contain melodic representations that are relatively similar to the corresponding regions of the classical reference space.
Musicological validation is required before converting these hypotheses into ground-truth annotations.

---

## 14. Scientific Interpretation

The experimental results reveal an important distinction between classification accuracy and structural music analysis.
The classical evaluation demonstrates that the Hybrid SVM can discriminate among 61 Raag classes with a Top-1 accuracy of 45.02% on the sealed TRF test set.

However, the Folk inference experiment demonstrates that classification confidence changes substantially under domain shift.
This suggests that a classical Raag classifier should not necessarily be interpreted as a universal Folk music classifier.
Instead, it can serve as a tool for identifying potential melodic affinities between Folk recordings and a classical Raag reference space.

This distinction is particularly important because Folk traditions may contain:
- mixed melodic grammars,
- regional performance practices,
- non-classical note usage,
- tonic variation,
- ornamentation,
- incomplete Raag structures,
- stylistic borrowing,
- and melodic forms that do not correspond to a single classical Raag.

---

## 15. Limitations

Several limitations must be acknowledged.

First, the classical training dataset contains only 635 recordings across 61 classes. The resulting low samples-per-class ratio limits the capacity of supervised models to learn highly generalizable class boundaries.
Second, the reported 45.02% accuracy represents forced-choice classification among the available 61 classes. It does not imply that every recording has an unambiguous Raag identity.
Third, the Folk predictions lack human-verified ground truth.
Fourth, the SARAGA evaluation used estimated-tonic evaluation and therefore should not be interpreted as perfectly equivalent to the TRF evaluation.
Fifth, pitch extraction and tonic normalization errors may propagate into the melodic representations.
Finally, the Folk confidence thresholds were derived from classical development data and therefore should be interpreted as indicators of model confidence rather than formally calibrated probabilities of musicological correctness.

---

## 16. Reproducibility and Data Integrity

The project followed a strict experimental isolation protocol.
The Folk dataset remained untouched throughout model development.
The TRF test set remained sealed until the final candidate satisfied the predefined development criteria.
Rejected approaches included dimensionality reduction, supervised metric learning, prototype-based retrieval, alternative aggregation strategies, and several tree-based classifiers.
The final model was frozen before Folk inference.
The resulting Folk dataset explicitly identifies predictions as computational hypotheses and separates model output from future human musicological validation.

---

## 17. Conclusion

This study presented a low-data Hybrid Geometric–Discriminative framework for Indian Raag classification.
The final representation combines melodic transitions, pitch-class distributions, and third-order melodic sequences within an 1,884-dimensional feature space.

A systematic experimental investigation demonstrated that several seemingly more sophisticated approaches were unable to generalize better than the domain-informed retrieval system. The strongest improvement was obtained by combining the geometric retrieval mechanism with an RBF-kernel SVM.
The final Hybrid SVM achieved 45.02% Top-1 accuracy on the sealed TRF test set and 56.60% estimated-tonic Top-1 performance on the external SARAGA evaluation.

The subsequent Folk inference demonstrated a substantial domain shift, with 86.5% of valid recordings falling into the predefined ambiguous confidence tier.
Rather than treating this ambiguity as a model failure, the study interprets it as evidence that classical Raag classification boundaries do not directly transfer to the full diversity of Indian Folk music.
The resulting Folk dataset should therefore be viewed as a collection of computational Raag-affinity hypotheses rather than ground-truth annotations.

The next scientific step is independent musicological validation of these hypotheses. Such validation could eventually transform the computational predictions into an expert-annotated dataset and provide a foundation for future research into the relationship between Indian Folk traditions and classical Raag frameworks.

---

## 18. Data and Code Availability

The project artifacts include the final Folk inference dataset, model configuration, feature definitions, inference outputs, integrity audits, confidence analyses, and documentation.
The published dataset should clearly distinguish:
- original recording metadata,
- computational features,
- model predictions,
- confidence measures,
- and future expert annotations.
Human-validated Raag labels should only be introduced after independent musicological review.

---

**Acknowledgements**
This work was conducted as part of an internship/research project investigating the Raag origins and structural affinities of Indian traditional Folk music.
The authors acknowledge the datasets and open-source resources used for classical music analysis and evaluation.

---

**Ethical and Scientific Statement**
The predicted Raag labels assigned to Folk recordings are computational hypotheses generated by a model trained on classical Raag data. They must not be interpreted as authoritative statements about the musical identity, cultural origin, or historical classification of any Folk tradition.
The project intentionally preserves this distinction to avoid converting statistical similarity into unsupported musicological claims.
