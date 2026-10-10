# RECOVERY AUDIT REPORT

## 1. DATASET AUDIT
- Total recordings: 49072
- Total duration (hours): 0.00
- Genres: 10 (Garba, Lavani, Dhamail, Sambalpuri, Bihu, Jhumur, Maand, Ghoomar, Baul, Bhangra)
- Audio formats: .wav, .mp3
- Failed/unreadable: 49072

## 2. EMBEDDINGS STATUS
- Recordings with embeddings: 626
- Total 5-second windows: 43142
- Recordings fully completed: 626 (assuming base file means complete for that recording)
- Recordings not processed: 48446
- NaN/Inf embeddings: 0
- Malformed embeddings: 0
- Completion percentage: 1.28%

## 3. SEQUENCE AND KNN STATUS
- 60-second sequences: 0
- Sequences evaluated (KNN): 0
- Recording predictions: 0
- Secondary BiLSTM: NOT COMPLETED
- Confidence Analysis: NOT COMPLETED

## 4. OVERALL CONCLUSION
- CASE C — PARTIAL EMBEDDING EXTRACTION

Minimum next action:
Resume CultureMERT extraction ONLY for the missing 48446 recordings. Do NOT reprocess completed recordings.
