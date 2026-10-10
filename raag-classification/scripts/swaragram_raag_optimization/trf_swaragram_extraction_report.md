# TRF Swaragram Extraction Report

## Summary
- **Recordings Processed:** 914
- **Recordings Skipped:** 189
- **Recordings Failed:** 0
- **Sequences Generated (60s chunks):** 11355
- **NaN Detected:** 0
- **Inf Detected:** 0

## Performance
- **Total Extraction Time:** 38.00 minutes
- **Average Processing Time per Recording:** 2.49 seconds

## Methodology
- 60-second non-overlapping contiguous chunks were extracted from each recording.
- Incomplete final chunks were discarded.
- Audio was resampled to 22,050 Hz and mono.
- Tonic normalization was applied on a per-recording basis (the tonic was estimated from the full audio recording).
- Train/Validation/Test boundaries were strictly preserved.
