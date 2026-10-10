# DATA SPLIT AND LEAKAGE AUDIT

- The inference pipeline uses distinct TRF reference data and Folk target data, meaning zero direct audio overlap (No Overlapping Segment Leakage).
- Singer-level grouping is unknown for Folk since metadata lacks strict performer identities.
- The model (CultureMERT) is pre-trained, preventing traditional train/val feature leakage.
