# Folk Music Dataset Duration Report

## 1. Overall Dataset Statistics
- **Total Songs**: 328
- **Total Cumulative Duration**: 113,185.57 seconds
- **Total Duration (HH:MM:SS)**: 31 hours, 26 minutes, 25 seconds
- **Average Song Duration**: 345.08 seconds (00:05:45)
- **Longest Recording**: `म्हारी सजनी ऐ आज तो बादिला घर आसी !! Rajasthani रज़वाड़ी Mand !! Singer- Nandlal Bhat !! लाईव ननावद.mp3` (1200.62 seconds, or ~20:00)
- **Shortest Recording**: `CHAHANEWALI _ New Sambalpuri song _ Song by Antara Chakrabarti and Ashish Kumbhar.mp3` (15.02 seconds)

## 2. Dataset Verification Status
- **Missing or Corrupted Files**: 0
- **Duplicate Audio Files Detected**: 0
- **Integrity Status**: Passed

## 3. Genre-wise Duration Statistics

| Genre | Number of Songs | Total Duration (HH:MM:SS) | Total Duration (Seconds) | Average Duration (MM:SS) |
| :--- | :--- | :--- | :--- | :--- |
| **Baul** | 30 | 02:24:30 | 8,670.48 sec | 04:49 |
| **Bhangra** | 32 | 02:35:36 | 9,336.82 sec | 04:51 |
| **Bihu** | 37 | 03:08:43 | 11,323.98 sec | 05:06 |
| **Dhamail** | 30 | 03:10:17 | 11,417.01 sec | 06:20 |
| **Garba** | 31 | 03:52:50 | 13,970.23 sec | 07:30 |
| **Ghoomar** | 30 | 03:02:20 | 10,940.49 sec | 06:04 |
| **Jhumur** | 30 | 02:35:01 | 9,301.85 sec | 05:10 |
| **Lavani** | 35 | 02:30:29 | 9,029.93 sec | 04:18 |
| **Maand** | 39 | 05:02:36 | 18,156.31 sec | 07:45 |
| **Sambalpuri**| 34 | 03:03:58 | 11,038.46 sec | 05:24 |

## Final Questions Answered Explicitly

1. **How many songs are present in the dataset?**
   There are exactly 328 distinct audio files in the raw dataset.
   
2. **What is the total cumulative duration of all songs?**
   113,185.57 seconds.
   
3. **What is the total duration in hours, minutes, and seconds?**
   31 hours, 26 minutes, and 25 seconds.
   
4. **What is the average song duration?**
   345.08 seconds (approximately 5 minutes and 45 seconds).
   
5. **Which song is the longest?**
   The longest song belongs to the Maand genre: `म्हारी सजनी ऐ आज तो बादिला घर आसी !! Rajasthani रज़वाड़ी Mand !! Singer- Nandlal Bhat !! लाईव ननावद.mp3`, lasting 1,200.62 seconds (about 20 minutes).
   
6. **Which song is the shortest?**
   The shortest song is `CHAHANEWALI _ New Sambalpuri song _ Song by Antara Chakrabarti and Ashish Kumbhar.mp3`, lasting 15.02 seconds.
   
7. **Were any corrupted or unreadable audio files found?**
   No. All 328 files were successfully parsed and read by the feature extraction library.
   
8. **Were any duplicate audio files detected?**
   No. An MD5 hash verification run during the duration calculation explicitly confirmed that there are exactly zero duplicate audio files across the entire dataset.
