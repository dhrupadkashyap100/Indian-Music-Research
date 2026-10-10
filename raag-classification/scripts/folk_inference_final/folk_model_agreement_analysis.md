# Model Agreement Diagnostics

## 1. Concept: Complementary Algorithms
The 45.02% Hybrid model consists of two radically different perspectives on the same acoustic representation:
1. **Geometric MAX Retrieval (75% weight):** Purely non-parametric. Calculates distance directly against original acoustic representations. It acts as an absolute similarity anchor.
2. **RBF-SVM Discriminator (25% weight):** Parametric and non-linear. Learns the warped boundaries and dimensional interactions that the raw cosine similarity ignores.

Model Agreement (when both Geometric and SVM Top-1 independently match the Hybrid result) serves as an alternative lens into prediction stability.

## 2. Agreement Statistics
- **Total Processed Recordings:** 327
- **Complete Agreement:** 54 instances
- **Disagreement:** 273 instances

The heavy disagreement rate is another massive indicator of OOD ambiguity. On TRF validation data, these models agree much more frequently. On Folk data, the SVM probability pushes back against the raw geometric similarity constantly.

## 3. Disagreement Domination
Because the model assigns a 75% weight to the Geometric score and only a 25% weight to the SVM, the Geometric model dominates the Hybrid outcome mathematically. However, because the SVM probabilities output very stark 1.0 vs 0.0 distributions (due to Platt scaling acting on out-of-domain data), there are many instances where the 25% SVM probability rescues the final ranking against a weak geometric preference.

## 4. Conclusion
Model agreement should be used in tandem with the Hybrid Margin. The 54 recordings featuring total agreement between the two fundamental algorithms represent the most structurally robust predictions generated during this research phase.
