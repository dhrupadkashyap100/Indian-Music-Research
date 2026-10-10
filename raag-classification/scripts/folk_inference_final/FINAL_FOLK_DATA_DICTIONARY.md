# Final Folk Raag Dataset Dictionary

This dictionary defines every column present in the `FINAL_FOLK_RAAG_DATASET.csv` master file.

| Column Name | Description / Meaning | Source / Type |
| :--- | :--- | :--- |
| `recording_id` | Immutable computational identifier used to connect predictions to raw audio. | Source Metadata |
| `audio_filename` | The filename of the audio track processed. | Source Metadata |
| `source_file_name` | Original source filename if distinct from audio_filename. | Source Metadata |
| `song_name` | The human-readable original song title, recovered from authoritative metadata (e.g. Baul_metadata.xlsx). | Source Metadata |
| `artist` | Artist performing the track, if authoritative metadata is available. | Source Metadata |
| `folk_genre` | The regional or stylistic genre of the Folk recording (e.g., Baul, Garba). | Source Metadata |
| `source` | External web source or recording origin (e.g., YouTube URL). | Source Metadata |
| `predicted_raag_top1` | **Model-generated hypothesis, NOT ground truth.** The classical Raag identified by the model as structurally most similar to the audio. | Model Generated |
| `predicted_raag_top2` | The second most probable classical Raag hypothesis. | Model Generated |
| `predicted_raag_top3` | The third most probable classical Raag hypothesis. | Model Generated |
| `hybrid_top1_score` | The final mathematical score (0.0 to 1.0) of the Top-1 hypothesis based on the 75/25 Hybrid fusion. | Model Generated |
| `hybrid_top2_score` | The final mathematical score (0.0 to 1.0) of the Top-2 hypothesis. | Model Generated |
| `hybrid_score_margin` | The difference between `hybrid_top1_score` and `hybrid_top2_score`. Serves as the primary indicator of ambiguity. | Model Generated |
| `confidence_tier` | Categorical confidence level (HIGH, MODERATE, AMBIGUOUS) mapped directly from the `hybrid_score_margin`. | Model Generated |
| `geometric_raag` | The raw Top-1 prediction produced exclusively by the 75% Non-parametric Cosine Similarity representation. | Model Generated |
| `svm_raag` | The raw Top-1 prediction produced exclusively by the 25% RBF-SVM discriminator. | Model Generated |
| `model_agreement` | Categorical evaluation of whether `geometric_raag` and `svm_raag` produced the identical hypothesis. | Model Generated |
| `review_priority` | Triage value (1.0, 2.0, 3.0) sorting recordings for future human musicologist validation. 1.0 represents highest structural confidence. | Model Generated |
| `inference_status` | Status flag confirming the audio successfully yielded feature representations (SUCCESS). | Model Generated |

## Note on Human Validation
The fields `musicologist_raag`, `validation_status`, and `musicologist_notes` are intentionally excluded from the master dataset but are provided as blank columns in `FOLK_RAAG_MUSICOLOGIST_REVIEW.csv`. These fields are strictly reserved for future human musicological validation and must not be populated algorithmically.
