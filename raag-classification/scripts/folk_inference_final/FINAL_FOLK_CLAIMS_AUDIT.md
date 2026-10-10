# Final Folk Claims Audit

This document subjects the project's conclusions to a strict scientific verification audit, categorizing each major claim regarding the Folk Music corpus.

## 1. SUPPORTED CLAIMS (Mathematically and Algorithmically Verified)

**Claim:** Indian Folk music exhibits profound Structural Ambiguity relative to classical grammar.
**Status:** SUPPORTED
**Evidence:** The calibrated Hybrid Score Margin triggered an `AMBIGUOUS` confidence flag for 86.5% of the recordings, meaning the out-of-domain acoustic features did not cluster neatly within a single rigid classical boundary.

**Claim:** The Hybrid Model is capable of detecting classical-like acoustic geometry within non-classical audio.
**Status:** SUPPORTED
**Evidence:** 54 recordings achieved absolute consensus between non-parametric geometric retrieval and non-linear SVM boundary classification, and 6 achieved margins comparable to verified classical recordings.

**Claim:** The algorithm mathematically associates Indian Folk music with structurally simpler classical Raags.
**Status:** SUPPORTED
**Evidence:** The distribution of the Top-5 predictions heavily concentrated on pentatonic/heptatonic Dhun-based scales (Malkauns, Pahadi, Durga), irrespective of genre.

## 2. PLAUSIBLE HYPOTHESES (Requiring Independent Human Verification)

**Claim:** Pahadi is the foundational structural backbone for a large percentage of northern Indian Folk music.
**Status:** PLAUSIBLE HYPOTHESIS
**Evidence:** The model predicted Pahadi or its direct structural neighbors (Suddha Kalyan/Durga) frequently. Musicologists historically assert this, and the model corroborated it, but without a human auditor confirming the specific 22 recordings, it remains an algorithmic hypothesis.

**Claim:** Model Disagreement (Geo vs. SVM) indicates the presence of "Mishra" (mixed-scale) phrasing or rhythmic audio distortion.
**Status:** PLAUSIBLE HYPOTHESIS
**Evidence:** The SVM is highly sensitive to overlapping decision boundaries, while the geometric model relies on absolute distance. Their divergence across 273 files strongly implies mixed classical signals, but requires auditory extraction to confirm.

**Claim:** The 6 `HIGH` confidence recordings are structurally identical to pure classical Raags.
**Status:** PLAUSIBLE HYPOTHESIS
**Evidence:** Their statistical margin matches verified TRF recordings, implying pure acoustic similarity.

## 3. NOT SUPPORTED (Invalid Scientific Extrapolations)

**Claim:** The algorithm proved that 9.2% of Indian Folk Music belongs to Raag Malkauns.
**Status:** NOT SUPPORTED
**Reason:** 61-class closed-set classification forces a choice. The model proved Malkauns was the *closest geometric match* available in its vocabulary, not the absolute musicological truth.

**Claim:** The Folk recordings with `AMBIGUOUS` confidence do not have a Raag.
**Status:** NOT SUPPORTED
**Reason:** The model is constrained to 61 classical classes. The ambiguity could stem from the Folk song utilizing a Raag outside the 61-class vocabulary, regional intonation differences, or poor acoustic F0 extraction due to heavy instrumentation. 

**Claim:** The model achieves roughly ~45% accuracy on Folk music because it achieved 45.02% on the Classical TRF Test set.
**Status:** NOT SUPPORTED
**Reason:** Accuracy metrics do not transfer across severe Domain Shifts. Folk music lacks ground truth; therefore, Folk accuracy is presently undefined and uncalculable.
