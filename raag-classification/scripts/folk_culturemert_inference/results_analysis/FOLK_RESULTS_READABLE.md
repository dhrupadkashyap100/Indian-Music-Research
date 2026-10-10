# Folk Raag Retrieval Results

## Overall Summary

- Total recordings: 328
- Successfully classified: 327
- Excluded: 1
- Total 60-second sequences: 2310
- Unique retrieved Raags: 53

## Raag Distribution

| Rank | Raag | Number of recordings | Percentage |
|---|---|---|---|
| 1 | Darbari | 28 | 8.56% |
| 2 | Adana | 23 | 7.03% |
| 3 | Marva | 15 | 4.59% |
| 4 | Khamaj | 15 | 4.59% |
| 5 | Bahar | 14 | 4.28% |
| 6 | Ahir Bhairav | 14 | 4.28% |
| 7 | Yaman Kalyan | 12 | 3.67% |
| 8 | Bhoopali | 11 | 3.36% |
| 9 | Bilaval | 11 | 3.36% |
| 10 | Asavari | 11 | 3.36% |
| 11 | Bhairavi | 10 | 3.06% |
| 12 | Purba | 10 | 3.06% |
| 13 | Megh | 10 | 3.06% |
| 14 | Malkauns | 9 | 2.75% |
| 15 | Desh | 9 | 2.75% |
| 16 | Alhaiya Bilaval | 9 | 2.75% |
| 17 | Bhairav | 8 | 2.45% |
| 18 | Bageshree | 8 | 2.45% |
| 19 | Sorath | 6 | 1.83% |
| 20 | Madhuvanti | 5 | 1.53% |
| 21 | Abhogi | 5 | 1.53% |
| 22 | Barwa | 5 | 1.53% |
| 23 | Nat Bhairav | 5 | 1.53% |
| 24 | Miyan Ki Todi orTodi or Darbari Todi | 5 | 1.53% |
| 25 | Yaman or Kalyan | 4 | 1.22% |
| 26 | Rageshree | 4 | 1.22% |
| 27 | suddha kalyan | 4 | 1.22% |
| 28 | Desi | 4 | 1.22% |
| 29 | Basant | 4 | 1.22% |
| 30 | Puriya | 3 | 0.92% |
| 31 | Gujari Todi | 3 | 0.92% |
| 32 | Shree | 3 | 0.92% |
| 33 | Basant Mukhari | 3 | 0.92% |
| 34 | Bilaskhani Todi | 3 | 0.92% |
| 35 | Bhatiyar | 3 | 0.92% |
| 36 | Lalit | 3 | 0.92% |
| 37 | Durga | 2 | 0.61% |
| 38 | Sohni | 2 | 0.61% |
| 39 | Megh Malhar | 2 | 0.61% |
| 40 | Jaunpuri | 2 | 0.61% |
| 41 | Bihag | 2 | 0.61% |
| 42 | Bhupal Todi | 2 | 0.61% |
| 43 | Kafi | 2 | 0.61% |
| 44 | Kedar | 2 | 0.61% |
| 45 | Malhar | 2 | 0.61% |
| 46 | Gauri | 2 | 0.61% |
| 47 | Pahadi | 2 | 0.61% |
| 48 | Bhimpalasi | 1 | 0.31% |
| 49 | Hindol | 1 | 0.31% |
| 50 | Poorvi | 1 | 0.31% |
| 51 | Deepak | 1 | 0.31% |
| 52 | Hemant | 1 | 0.31% |
| 53 | Komal Bhairav | 1 | 0.31% |

## Genre-wise Results

| Genre | # Recordings | Top-1 Raag | % | 2nd | 3rd |
|---|---|---|---|---|---|
| Baul | 30 | Yaman Kalyan | 13.3% | Bhairavi | Purba |
| Bhangra | 32 | Darbari | 28.1% | Adana | Gujari Todi |
| Bihu | 37 | Bhoopali | 13.5% | Desh | Ahir Bhairav |
| Dhamail | 30 | Darbari | 16.7% | Khamaj | Adana |
| Garba | 31 | Adana | 12.9% | Megh | Alhaiya Bilaval |
| Ghoomar | 30 | Bhairav | 20.0% | Adana | Asavari |
| Jhumur | 30 | Bageshree | 13.3% | Adana | Marva |
| Lavani | 35 | Bahar | 14.3% | Marva | suddha kalyan |
| Maand | 39 | Darbari | 10.3% | Purba | Khamaj |
| Sambalpuri | 33 | Bilaval | 9.1% | Sorath | Darbari |

## Data Observations

**DATA OBSERVATION**: The Raag *Darbari* appears strongly in Bhangra, while *Adana* is dominant in Ghoomar. These distinct Classical Raags are consistently retrieved by specific Folk genres.
**DATA OBSERVATION**: A large variety of Raags (53 unique) were retrieved across the 327 recordings, indicating that the Folk dataset spans a diverse acoustic/melodic space compared to the TRF references.
**DATA OBSERVATION**: Several high-margin recordings have similarities > 0.8, showing surprisingly strong alignment with specific classical references despite the domain gap.
**HYPOTHESIS**: Genres with concentrated retrievals (e.g., Bhangra with Darbari) might share foundational scale intervals (thaats) or distinctive melodic movements with those Classical Raags, triggering strong CultureMERT alignment.

## 20 Highest-Margin Recordings

**Song:** Manush Bhajo Manusheyi Pujo (BAUL_023)
**Genre:** Baul
1. Purba — similarity: 0.9455
2. nan — similarity: 0.0000
3. nan — similarity: N/A (Only top 2 sim stored in csv)
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.9455

**Song:** Jay Aadhyashakti (GARBA_023)
**Genre:** Garba
1. Purba — similarity: 0.9348
2. nan — similarity: 0.0000
3. nan — similarity: N/A (Only top 2 sim stored in csv)
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.9348

**Song:** Run Jhun Baje Ghooghara (GHOOMAR_019)
**Genre:** Ghoomar
1. Alhaiya Bilaval — similarity: 0.9360
2. suddha kalyan — similarity: 0.9205
3. Pahadi — similarity: N/A (Only top 2 sim stored in csv)
4. Bhimpalasi — similarity: N/A
5. Adana — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0156

**Song:** Tunak Tunak Tun (BHANGRA_007)
**Genre:** Bhangra
1. Megh — similarity: 0.9487
2. Deepak — similarity: 0.9359
3. Bhatiyar — similarity: N/A (Only top 2 sim stored in csv)
4. Bilaskhani Todi — similarity: N/A
5. sarang — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0127

**Song:** জলের ঘাটে (DHAMAIL_021)
**Genre:** Dhamail
1. Bihag — similarity: 0.9414
2. Yaman Kalyan — similarity: 0.9299
3. Kedar — similarity: N/A (Only top 2 sim stored in csv)
4. Nat Bhairav — similarity: N/A
5. Darbari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0115

**Song:** Mor Dhoni Go (JHUMUR_0012)
**Genre:** Jhumur
1. Malkauns — similarity: 0.9293
2. Madhuvanti — similarity: 0.9192
3. Bilaval — similarity: N/A (Only top 2 sim stored in csv)
4. Asavari — similarity: N/A
5. Sorath — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0101

**Song:** Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) (BHANGRA_031)
**Genre:** Bhangra
1. Rageshree — similarity: 0.9379
2. Khamaj — similarity: 0.9289
3. Bilaval — similarity: N/A (Only top 2 sim stored in csv)
4. Sorath — similarity: N/A
5. Bhairavi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0090

**Song:** Dil Dariyar Majhete (BAUL_024)
**Genre:** Baul
1. Purba — similarity: 0.9334
2. Bageshree — similarity: 0.9246
3. Desh — similarity: N/A (Only top 2 sim stored in csv)
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0088

**Song:** Kashyala Lavato (LAVANI_017)
**Genre:** Lavani
1. Barwa — similarity: 0.9223
2. Darbari — similarity: 0.9137
3. Alhaiya Bilaval — similarity: N/A (Only top 2 sim stored in csv)
4. Ahir Bhairav — similarity: N/A
5. Kafi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0086

**Song:** Naki Doli Chhaan (LAVANI_012)
**Genre:** Lavani
1. Puriya — similarity: 0.9578
2. Kedar — similarity: 0.9494
3. Bageshree — similarity: N/A (Only top 2 sim stored in csv)
4. Megh — similarity: N/A
5. Bihag — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0084

**Song:** Prem Koraa Ki Jwalaa (BAUL_021)
**Genre:** Baul
1. Purba — similarity: 0.9417
2. Gauri — similarity: 0.9334
3. Bageshree — similarity: N/A (Only top 2 sim stored in csv)
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0084

**Song:** O Dhei Oi (BIHU_019)
**Genre:** Bihu
1. Megh — similarity: 0.9416
2. Bhairavi — similarity: 0.9335
3. Desh — similarity: N/A (Only top 2 sim stored in csv)
4. Bahar — similarity: N/A
5. Bageshree — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0081

**Song:** Apna Sangeet Vajeh Apna Sangeet (BHANGRA_024)
**Genre:** Bhangra
1. Bilaval — similarity: 0.9226
2. Desh — similarity: 0.9145
3. Ahir Bhairav — similarity: N/A (Only top 2 sim stored in csv)
4. Bageshree — similarity: N/A
5. Darbari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0081

**Song:** Ambran Tu Ayi Hui Hoor Sohniye (BHANGRA_014)
**Genre:** Bhangra
1. Gujari Todi — similarity: 0.9384
2. Megh — similarity: 0.9304
3. Puriya — similarity: N/A (Only top 2 sim stored in csv)
4. Bilaskhani Todi — similarity: N/A
5. Miyan Ki Todi orTodi or Darbari Todi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0081

**Song:** Badila Yahi Rahiyo Saa (MAAND_010)
**Genre:** Maand
1. Gauri — similarity: 0.9537
2. Bageshree — similarity: 0.9457
3. Adana — similarity: N/A (Only top 2 sim stored in csv)
4. Desh — similarity: N/A
5. Khamaj — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0080

**Song:** Chor Poreche Babur Bagane (BAUL_011)
**Genre:** Baul
1. Bhairavi — similarity: 0.9281
2. Megh — similarity: 0.9205
3. Gauri — similarity: N/A (Only top 2 sim stored in csv)
4. Alhaiya Bilaval — similarity: N/A
5. Abhogi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0076

**Song:** Rasia (SAMBALPURI_006)
**Genre:** Sambalpuri
1. Bilaval — similarity: 0.9351
2. Bageshree — similarity: 0.9276
3. Bhimpalasi — similarity: N/A (Only top 2 sim stored in csv)
4. Asavari — similarity: N/A
5. Desh — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0074

**Song:** Ho Jayegi Balle Balle (BHANGRA_009)
**Genre:** Bhangra
1. Darbari — similarity: 0.9468
2. Basant — similarity: 0.9397
3. Asavari — similarity: N/A (Only top 2 sim stored in csv)
4. Kedar — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0071

**Song:** চান্নি রাইতে আইও কালা (DHAMAIL_015)
**Genre:** Dhamail
1. Madhuvanti — similarity: 0.9452
2. Miyan Ki Todi orTodi or Darbari Todi — similarity: 0.9382
3. Multani — similarity: N/A (Only top 2 sim stored in csv)
4. Nat Bhairav — similarity: N/A
5. Asavari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0070

**Song:** Kulfirani Chocobar (SAMBALPURI_028)
**Genre:** Sambalpuri
1. Marva — similarity: 0.9496
2. Megh — similarity: 0.9429
3. Asavari — similarity: N/A (Only top 2 sim stored in csv)
4. Alhaiya Bilaval — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0067

## 20 Lowest-Margin Recordings

**Song:** Putt Jattan De (BHANGRA_029)
**Genre:** Bhangra
1. Abhogi — similarity: 0.9170
2. Darbari — similarity: 0.9168
3. Pahadi — similarity: N/A
4. Bhairavi — similarity: N/A
5. Desi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0002

**Song:** Hae Go (SAMBALPURI_008)
**Genre:** Sambalpuri
1. Darbari — similarity: 0.9347
2. Megh — similarity: 0.9345
3. Sorath — similarity: N/A
4. Bilaval — similarity: N/A
5. Ahir Bhairav — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0002

**Song:** Sonamoni (JHUMUR_0019)
**Genre:** Jhumur
1. Adana — similarity: 0.9368
2. Desh — similarity: 0.9367
3. Sorath — similarity: N/A
4. Bahar — similarity: N/A
5. Barwa — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0002

**Song:** Ghotala Ghotala Zala (LAVANI_030)
**Genre:** Lavani
1. Bilaskhani Todi — similarity: 0.9362
2. Marva — similarity: 0.9360
3. Desh — similarity: N/A
4. Bageshree — similarity: N/A
5. Puriya — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Tera Yaar Bolda (BHANGRA_001)
**Genre:** Bhangra
1. Bahar — similarity: 0.9212
2. Darbari — similarity: 0.9210
3. Alhaiya Bilaval — similarity: N/A
4. Bibhas — similarity: N/A
5. Bhoopali — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** যুগল মিলন হইলো গো (DHAMAIL_016)
**Genre:** Dhamail
1. Bihag — similarity: 0.9409
2. Kedar — similarity: 0.9408
3. Multani — similarity: N/A
4. Bhimpalasi — similarity: N/A
5. Miyan Ki Todi orTodi or Darbari Todi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Ramagaru (SAMBALPURI_013)
**Genre:** Sambalpuri
1. Sorath — similarity: 0.9276
2. Desh — similarity: 0.9275
3. Alhaiya Bilaval — similarity: N/A
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Ki Sai Aso (BIHU_013)
**Genre:** Bihu
1. Ahir Bhairav — similarity: 0.9378
2. Asavari — similarity: 0.9377
3. Adana — similarity: N/A
4. Bhoopali — similarity: N/A
5. Puriya — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Bai Mazi Karangali Modali (LAVANI_026)
**Genre:** Lavani
1. Basant Mukhari — similarity: 0.9551
2. Puriya — similarity: 0.9550
3. Bihag — similarity: N/A
4. Rageshree — similarity: N/A
5. Darbari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Din Gani Gani (SAMBALPURI_033)
**Genre:** Sambalpuri
1. Shree — similarity: 0.9340
2. Alhaiya Bilaval — similarity: 0.9339
3. Khamaj — similarity: N/A
4. Desi — similarity: N/A
5. Asavari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Aami Ek Pagal Baul (BAUL_019)
**Genre:** Baul
1. Bhairavi — similarity: 0.9208
2. Ahir Bhairav — similarity: 0.9207
3. Sorath — similarity: N/A
4. Nat Bhairav — similarity: N/A
5. Darbari — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Chhap Tilak Sab Cheeni (MAAND_022)
**Genre:** Maand
1. Adana — similarity: 0.9323
2. Bahar — similarity: 0.9323
3. Darbari — similarity: N/A
4. Nat Bhairav — similarity: N/A
5. suddha kalyan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0001

**Song:** Raag Maand Bandish (MAAND_019)
**Genre:** Maand
1. Bilaval — similarity: 0.9388
2. Purba — similarity: 0.9388
3. Sorath — similarity: N/A
4. Khamaj — similarity: N/A
5. Adana — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Dupatta Tera Satrang Da (BHANGRA_003)
**Genre:** Bhangra
1. Darbari — similarity: 0.9340
2. Alhaiya Bilaval — similarity: 0.9340
3. Bibhas — similarity: N/A
4. Bahar — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Koliye Juge Alai Doyal (JHUMUR_0023)
**Genre:** Jhumur
1. Lalit — similarity: 0.9528
2. Megh — similarity: 0.9527
3. Adana — similarity: N/A
4. nan — similarity: N/A
5. nan — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Laal Pari (JHUMUR_0024)
**Genre:** Jhumur
1. Madhuvanti — similarity: 0.9358
2. Sorath — similarity: 0.9358
3. Desh — similarity: N/A
4. Megh — similarity: N/A
5. Adana — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Nistei (JHUMUR_002)
**Genre:** Jhumur
1. Bageshree — similarity: 0.9418
2. Sohni — similarity: 0.9418
3. Pahadi — similarity: N/A
4. Marva — similarity: N/A
5. Sorath — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Tumake Logot (BIHU_004)
**Genre:** Bihu
1. Bilaval — similarity: 0.9349
2. Puriya — similarity: 0.9349
3. Malkauns — similarity: N/A
4. Darbari — similarity: N/A
5. Abhogi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Jhuli Jhule (JHUMUR_0013)
**Genre:** Jhumur
1. Desi — similarity: 0.9251
2. Madhuvanti — similarity: 0.9251
3. Khamaj — similarity: N/A
4. Marva — similarity: N/A
5. Ahir Bhairav — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

**Song:** Petore Juruni Bhat (BIHU_021)
**Genre:** Bihu
1. Megh Malhar — similarity: 0.9354
2. Malhar — similarity: 0.9354
3. Adana — similarity: N/A
4. Bilaval — similarity: N/A
5. Bhairavi — similarity: N/A
**Margin:** Top-1 − Top-2 = 0.0000

## Complete 327-Recording Prediction Table

| Original Song ID | Song Title | Artist | Genre | Top-1 Raag | Top-1 Sim | Top-2 Raag | Top-2 Sim | Margin | Top-3 Candidates | Top-5 Candidates |
|---|---|---|---|---|---|---|---|---|---|---|
| BAUL_001 | Agun Pani | Purna Das Baul | Baul | Madhuvanti | 0.9390 | Darbari | 0.9383 | 0.0008 | Madhuvanti, Darbari, Bhairavi | Madhuvanti, Darbari, Bhairavi, Purba, Bahar |
| BAUL_002 | Ja Lolite Anga Kushum Tule | Purna Das Baul | Baul | Abhogi | 0.9225 | Malkauns | 0.9201 | 0.0025 | Abhogi, Malkauns, Darbari | Abhogi, Malkauns, Darbari, Yaman Kalyan, Alhaiya Bilaval |
| BAUL_003 | Kalkatou Mon Heshekeley | Purna Das Baul | Baul | Yaman Kalyan | 0.9422 | Malkauns | 0.9393 | 0.0029 | Yaman Kalyan, Malkauns, Abhogi | Yaman Kalyan, Malkauns, Abhogi, Madhuvanti, Darbari |
| BAUL_004 | Ananda Lahari | Purna Das Baul | Baul | Marva | 0.9303 | Bhairavi | 0.9264 | 0.0039 | Marva, Bhairavi, Khamaj | Marva, Bhairavi, Khamaj, Malhar, Darbari |
| BAUL_005 | Amar Aei Rudir Dhara | Purna Das Baul | Baul | Bhairavi | 0.9324 | Alhaiya Bilaval | 0.9293 | 0.0031 | Bhairavi, Alhaiya Bilaval, Malkauns | Bhairavi, Alhaiya Bilaval, Malkauns, Purba, sarang |
| BAUL_006 | Hari Balo Re | Purna Das Baul | Baul | Yaman Kalyan | 0.9325 | Malkauns | 0.9291 | 0.0033 | Yaman Kalyan, Malkauns, Puriya | Yaman Kalyan, Malkauns, Puriya, Darbari, Sorath |
| BAUL_007 | Gour Lila Bazare | Purna Das Baul | Baul | Yaman Kalyan | 0.9373 | Megh | 0.9314 | 0.0059 | Yaman Kalyan, Megh, Darbari | Yaman Kalyan, Megh, Darbari, Yaman or Kalyan, Kafi |
| BAUL_008 | Tripti | Purna Das Baul | Baul | Malkauns | 0.8812 | Sorath | 0.8810 | 0.0002 | Malkauns, Sorath, Basant | Malkauns, Sorath, Basant, Bhairavi, Malhar |
| BAUL_009 | Kon Ba Zharir | Purna Das Baul | Baul | Barwa | 0.9362 | Lalit | 0.9350 | 0.0012 | Barwa, Lalit, Pahadi | Barwa, Lalit, Pahadi, Desh, Puriya |
| BAUL_010 | Sonogo Aian Dada | Purna Das Baul | Baul | Yaman Kalyan | 0.9457 | Bahar | 0.9417 | 0.0040 | Yaman Kalyan, Bahar, Malkauns | Yaman Kalyan, Bahar, Malkauns, Bhimpalasi, Asavari |
| BAUL_011 | Chor Poreche Babur Bagane | Purna Das Baul | Baul | Bhairavi | 0.9281 | Megh | 0.9205 | 0.0076 | Bhairavi, Megh, Gauri | Bhairavi, Megh, Gauri, Alhaiya Bilaval, Abhogi |
| BAUL_012 | Golemale Gole Male | Purna Das Baul | Baul | Khamaj | 0.9357 | Jhinjhoti | 0.9341 | 0.0016 | Khamaj, Jhinjhoti, Abhogi | Khamaj, Jhinjhoti, Abhogi, Bilaval, Purba |
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Purna Das Baul | Baul | Puriya | 0.9328 | Jaunpuri | 0.9276 | 0.0052 | Puriya, Jaunpuri, Pahadi | Puriya, Jaunpuri, Pahadi, Sohni |
| BAUL_014 | Dekhechhi Rup-Sagare | Purna Das Baul | Baul | Bhairavi | 0.9367 | Sorath | 0.9317 | 0.0049 | Bhairavi, Sorath, nan | Bhairavi, Sorath |
| BAUL_015 | Aami Kothay Pabo Tare | Purna Das Baul | Baul | Abhogi | 0.9181 | Bhairavi | 0.9172 | 0.0009 | Abhogi, Bhairavi, Malkauns | Abhogi, Bhairavi, Malkauns, Asavari, Kedar |
| BAUL_016 | Shesher Deene Sejon Bine | Purna Das Baul | Baul | Sorath | 0.9355 | Madhuvanti | 0.9304 | 0.0051 | Sorath, Madhuvanti, Darbari | Sorath, Madhuvanti, Darbari, Malkauns, Purba |
| BAUL_017 | Milon Hobe Koto Dine | Farida Parveen | Baul | Bhimpalasi | 0.9335 | Bhoopali | 0.9318 | 0.0017 | Bhimpalasi, Bhoopali, Barwa | Bhimpalasi, Bhoopali, Barwa, Desh, Pahadi |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Farida Parveen | Baul | Abhogi | 0.9294 | Yaman or Kalyan | 0.9289 | 0.0005 | Abhogi, Yaman or Kalyan, Kafi | Abhogi, Yaman or Kalyan, Kafi, Jhinjhoti, Adana |
| BAUL_019 | Aami Ek Pagal Baul | Purna Das Baul | Baul | Bhairavi | 0.9208 | Ahir Bhairav | 0.9207 | 0.0001 | Bhairavi, Ahir Bhairav, Sorath | Bhairavi, Ahir Bhairav, Sorath, Nat Bhairav, Darbari |
| BAUL_020 | Jemon Beni Temni Rabey | Purna Das Baul | Baul | Bhairav | 0.9400 | Ramkali | 0.9357 | 0.0043 | Bhairav, Ramkali, Khamaj | Bhairav, Ramkali, Khamaj, Desh, Megh Malhar |
| BAUL_021 | Prem Koraa Ki Jwalaa | Purna Das Baul | Baul | Purba | 0.9417 | Gauri | 0.9334 | 0.0084 | Purba, Gauri, Bageshree | Purba, Gauri, Bageshree |
| BAUL_022 | Emon Mon Byabsaa | Purna Das Baul | Baul | Durga | 0.9488 | Gauri | 0.9449 | 0.0040 | Durga, Gauri, Rageshree | Durga, Gauri, Rageshree, Asavari, Bihag |
| BAUL_023 | Manush Bhajo Manusheyi Pujo | Purna Das Baul | Baul | Purba | 0.9455 | nan | 0.0000 | 0.9455 | Purba, nan, nan | Purba |
| BAUL_024 | Dil Dariyar Majhete | Purna Das Baul | Baul | Purba | 0.9334 | Bageshree | 0.9246 | 0.0088 | Purba, Bageshree, Desh | Purba, Bageshree, Desh |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Purna Das Baul | Baul | Purba | 0.9353 | Darbari | 0.9294 | 0.0059 | Purba, Darbari, Rageshree | Purba, Darbari, Rageshree, Kedar |
| BAUL_026 | Sokhi Jamuna Jol | Purna Das Baul | Baul | Malkauns | 0.9320 | Megh | 0.9315 | 0.0005 | Malkauns, Megh, Yaman or Kalyan | Malkauns, Megh, Yaman or Kalyan, Yaman Kalyan, Desh |
| BAUL_027 | Pre Loye Jao Amay | Farida Parveen | Baul | Malkauns | 0.9434 | Bhairav | 0.9402 | 0.0032 | Malkauns, Bhairav, Poorvi | Malkauns, Bhairav, Poorvi, Abhogi, Bhoopali |
| BAUL_028 | Somoy Gele Sadhon Hobena | Farida Parveen | Baul | Nat Bhairav | 0.9180 | Megh | 0.9171 | 0.0009 | Nat Bhairav, Megh, Bageshree | Nat Bhairav, Megh, Bageshree, Khamaj |
| BAUL_029 | Aar Amare Marishne Maa | Farida Parveen | Baul | Miyan Ki Todi orTodi or Darbari Todi | 0.9414 | Bahar | 0.9394 | 0.0020 | Miyan Ki Todi orTodi or Darbari Todi, Bahar, Desh | Miyan Ki Todi orTodi or Darbari Todi, Bahar, Desh, Bilaskhani Todi, Pahadi |
| BAUL_030 | Dhonno Dhonno Boli Tare | Farida Parveen | Baul | Bahar | 0.9370 | Miyan Ki Todi orTodi or Darbari Todi | 0.9339 | 0.0031 | Bahar, Miyan Ki Todi orTodi or Darbari Todi, Bihag | Bahar, Miyan Ki Todi orTodi or Darbari Todi, Bihag, Bageshree, Yaman Kalyan |
| BHANGRA_001 | Tera Yaar Bolda | Surjit Bindrakhia | Bhangra | Bahar | 0.9212 | Darbari | 0.9210 | 0.0001 | Bahar, Darbari, Alhaiya Bilaval | Bahar, Darbari, Alhaiya Bilaval, Bibhas, Bhoopali |
| BHANGRA_002 | Jatt Di Pasand | Surjit Bindrakhia | Bhangra | Bhoopali | 0.9307 | Bahar | 0.9272 | 0.0035 | Bhoopali, Bahar, Asavari | Bhoopali, Bahar, Asavari, Darbari, Yaman or Kalyan |
| BHANGRA_003 | Dupatta Tera Satrang Da | Surjit Bindrakhia | Bhangra | Darbari | 0.9340 | Alhaiya Bilaval | 0.9340 | 0.0000 | Darbari, Alhaiya Bilaval, Bibhas | Darbari, Alhaiya Bilaval, Bibhas, Bahar |
| BHANGRA_004 | Wang Wargi Kudi | Surjit Bindrakhia | Bhangra | Darbari | 0.9286 | Abhogi | 0.9243 | 0.0043 | Darbari, Abhogi, Ahir Bhairav | Darbari, Abhogi, Ahir Bhairav |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Surjit Bindrakhia | Bhangra | Yaman or Kalyan | 0.9279 | Megh | 0.9239 | 0.0041 | Yaman or Kalyan, Megh, Bahar | Yaman or Kalyan, Megh, Bahar, Khamaj, Marva |
| BHANGRA_006 | Munda Ki Mangda | Surjit Bindrakhia | Bhangra | Adana | 0.9311 | Alhaiya Bilaval | 0.9282 | 0.0029 | Adana, Alhaiya Bilaval, Bhairav Bahar | Adana, Alhaiya Bilaval, Bhairav Bahar, Jaunpuri |
| BHANGRA_007 | Tunak Tunak Tun | Daler Mehndi | Bhangra | Megh | 0.9487 | Deepak | 0.9359 | 0.0127 | Megh, Deepak, Bhatiyar | Megh, Deepak, Bhatiyar, Bilaskhani Todi, sarang |
| BHANGRA_008 | Bolo Ta Ra Ra | Daler Mehndi | Bhangra | Desh | 0.9417 | Pahadi | 0.9385 | 0.0032 | Desh, Pahadi, Asavari | Desh, Pahadi, Asavari, Miyan Ki Todi orTodi or Darbari Todi |
| BHANGRA_009 | Ho Jayegi Balle Balle | Daler Mehndi | Bhangra | Darbari | 0.9468 | Basant | 0.9397 | 0.0071 | Darbari, Basant, Asavari | Darbari, Basant, Asavari, Kedar |
| BHANGRA_010 | Dardi Rab Rab Kardi | Daler Mehndi | Bhangra | Darbari | 0.9320 | Asavari | 0.9275 | 0.0045 | Darbari, Asavari, Alhaiya Bilaval | Darbari, Asavari, Alhaiya Bilaval, Miyan Ki Todi orTodi or Darbari Todi |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Daler Mehndi | Bhangra | Adana | 0.9344 | Ahir Bhairav | 0.9298 | 0.0046 | Adana, Ahir Bhairav, Barwa | Adana, Ahir Bhairav, Barwa, Deepak, Bibhas |
| BHANGRA_012 | Na Na Na Na Na Re | Daler Mehndi | Bhangra | Adana | 0.9397 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | 0.0015 | Adana, Miyan Ki Todi orTodi or Darbari Todi, Madhuvanti | Adana, Miyan Ki Todi orTodi or Darbari Todi, Madhuvanti, Rageshree, Alhaiya Bilaval |
| BHANGRA_013 | Pao Bhangra | Balwinder Safri | Bhangra | Bhairavi | 0.9151 | Bahar | 0.9140 | 0.0011 | Bhairavi, Bahar, Malhar | Bhairavi, Bahar, Malhar, Megh Malhar, Asavari |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Balwinder Safri | Bhangra | Gujari Todi | 0.9384 | Megh | 0.9304 | 0.0081 | Gujari Todi, Megh, Puriya | Gujari Todi, Megh, Puriya, Bilaskhani Todi, Miyan Ki Todi orTodi or Darbari Todi |
| BHANGRA_015 | Sariyan Toon Sohni | Balwinder Safri | Bhangra | Alhaiya Bilaval | 0.9291 | Darbari | 0.9286 | 0.0005 | Alhaiya Bilaval, Darbari, Asavari | Alhaiya Bilaval, Darbari, Asavari, Puriya, Ahir Bhairav |
| BHANGRA_016 | Mitran Ne Mundri Te | Balwinder Safri | Bhangra | Darbari | 0.9400 | Puriya | 0.9394 | 0.0006 | Darbari, Puriya, Alhaiya Bilaval | Darbari, Puriya, Alhaiya Bilaval, Asavari |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Balwinder Safri | Bhangra | Gujari Todi | 0.9387 | Bilaval | 0.9333 | 0.0054 | Gujari Todi, Bilaval, Puriya | Gujari Todi, Bilaval, Puriya, Megh, Alhaiya Bilaval |
| BHANGRA_018 | Tutak Tutak Tutiya | Malkit Singh (Golden Star) | Bhangra | Darbari | 0.9297 | Alhaiya Bilaval | 0.9295 | 0.0002 | Darbari, Alhaiya Bilaval, Desh | Darbari, Alhaiya Bilaval, Desh, Asavari, Gujari Todi |
| BHANGRA_019 | Kurri Garam Jayee | Malkit Singh (Golden Star) | Bhangra | Rageshree | 0.9322 | Puriya | 0.9319 | 0.0003 | Rageshree, Puriya, Gauri | Rageshree, Puriya, Gauri, Kedar |
| BHANGRA_020 | Chal Hun | Malkit Singh (Golden Star) | Bhangra | Shree | 0.9341 | Gujari Todi | 0.9327 | 0.0014 | Shree, Gujari Todi, Bhimpalasi | Shree, Gujari Todi, Bhimpalasi, Darbari, Madhuvanti |
| BHANGRA_021 | Kinni Sohni | Malkit Singh (Golden Star) | Bhangra | Darbari | 0.9421 | Alhaiya Bilaval | 0.9358 | 0.0063 | Darbari, Alhaiya Bilaval, Rageshree | Darbari, Alhaiya Bilaval, Rageshree, Adana, Desi |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Apna Sangeet | Bhangra | Bilaval | 0.9433 | Malkauns | 0.9424 | 0.0009 | Bilaval, Malkauns, Yaman Kalyan | Bilaval, Malkauns, Yaman Kalyan, Sorath, Purba |
| BHANGRA_023 | Bhaina Nachiyaan | Apna Sangeet | Bhangra | Alhaiya Bilaval | 0.9301 | Puriya | 0.9252 | 0.0049 | Alhaiya Bilaval, Puriya, Asavari | Alhaiya Bilaval, Puriya, Asavari, Barwa |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Apna Sangeet | Bhangra | Bilaval | 0.9226 | Desh | 0.9145 | 0.0081 | Bilaval, Desh, Ahir Bhairav | Bilaval, Desh, Ahir Bhairav, Bageshree, Darbari |
| BHANGRA_025 | Soho Road Utey | Apna Sangeet | Bhangra | Malkauns | 0.9347 | Bilaval | 0.9321 | 0.0026 | Malkauns, Bilaval, Desh | Malkauns, Bilaval, Desh, Darbari, Khamaj |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Kuldeep Manak | Bhangra | Darbari | 0.9309 | Asavari | 0.9244 | 0.0065 | Darbari, Asavari, Alhaiya Bilaval | Darbari, Asavari, Alhaiya Bilaval, Madhuvanti |
| BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Kuldeep Manak | Bhangra | Darbari | 0.9342 | Asavari | 0.9312 | 0.0030 | Darbari, Asavari, nan | Darbari, Asavari |
| BHANGRA_028 | Jugni | Kuldeep Manak | Bhangra | Yaman Kalyan | 0.9351 | Sorath | 0.9308 | 0.0043 | Yaman Kalyan, Sorath, Adana | Yaman Kalyan, Sorath, Adana, Bhimpalasi, Desh |
| BHANGRA_029 | Putt Jattan De | Surinder Shinda | Bhangra | Abhogi | 0.9170 | Darbari | 0.9168 | 0.0002 | Abhogi, Darbari, Pahadi | Abhogi, Darbari, Pahadi, Bhairavi, Desi |
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Surinder Shinda | Bhangra | suddha kalyan | 0.9264 | Purba | 0.9229 | 0.0035 | suddha kalyan, Purba, Adana | suddha kalyan, Purba, Adana, Bilaskhani Todi |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Surinder Shinda | Bhangra | Rageshree | 0.9379 | Khamaj | 0.9289 | 0.0090 | Rageshree, Khamaj, Bilaval | Rageshree, Khamaj, Bilaval, Sorath, Bhairavi |
| BHANGRA_032 | Jeona Morh | Surinder Shinda | Bhangra | Basant Mukhari | 0.9448 | Bhairav | 0.9441 | 0.0007 | Basant Mukhari, Bhairav, Basant | Basant Mukhari, Bhairav, Basant, Megh Malhar, Malhar |
| BIHU_001 | Morom Logai Logai | Zubeen Garg | Bihu | Sohni | 0.9355 | Bahar | 0.9334 | 0.0021 | Sohni, Bahar, Megh Malhar | Sohni, Bahar, Megh Malhar, Bhimpalasi, suddha kalyan |
| BIHU_002 | Toi Nagini Ne | Zubeen Garg | Bihu | Asavari | 0.9280 | Puriya | 0.9271 | 0.0009 | Asavari, Puriya, Sohni | Asavari, Puriya, Sohni, Bahar, Khamaj |
| BIHU_003 | Kokalote Tongali | Zubeen Garg | Bihu | Hindol | 0.9402 | Sorath | 0.9400 | 0.0002 | Hindol, Sorath, Madhuvanti | Hindol, Sorath, Madhuvanti, suddha kalyan, Khamaj |
| BIHU_004 | Tumake Logot | Zubeen Garg | Bihu | Bilaval | 0.9349 | Puriya | 0.9349 | 0.0000 | Bilaval, Puriya, Malkauns | Bilaval, Puriya, Malkauns, Darbari, Abhogi |
| BIHU_005 | Kinu Sawonire Sala | Zubeen Garg | Bihu | Desi | 0.9378 | Darbari | 0.9354 | 0.0024 | Desi, Darbari, Hindol | Desi, Darbari, Hindol, Desh, Ahir Bhairav |
| BIHU_006 | Dikhow Noi Eribo | Zubeen Garg | Bihu | Desh | 0.9435 | Sorath | 0.9399 | 0.0036 | Desh, Sorath, Marva | Desh, Sorath, Marva |
| BIHU_007 | Phool Phulise | Zubeen Garg | Bihu | Desh | 0.9279 | Pahadi | 0.9243 | 0.0037 | Desh, Pahadi, Malhar | Desh, Pahadi, Malhar, Darbari, Desi |
| BIHU_008 | Porbotor Dhekiya | Zubeen Garg | Bihu | Desh | 0.9392 | Desi | 0.9379 | 0.0013 | Desh, Desi, Nat Bhairav | Desh, Desi, Nat Bhairav, Madhuvanti, Barwa |
| BIHU_009 | O Nahor | Zubeen Garg | Bihu | Marva | 0.9373 | Bahar | 0.9318 | 0.0055 | Marva, Bahar, Puriya | Marva, Bahar, Puriya, Bhimpalasi, Ahir Bhairav |
| BIHU_010 | Janmoni Xopunote Ahiba | Zubeen Garg | Bihu | Bilaskhani Todi | 0.9465 | Purba | 0.9454 | 0.0010 | Bilaskhani Todi, Purba, Hindol | Bilaskhani Todi, Purba, Hindol, Asavari, Desh |
| BIHU_011 | Eijon Korenu Nasoni | Bhupen Hazarika | Bihu | Adana | 0.9127 | Bhimpalasi | 0.9113 | 0.0014 | Adana, Bhimpalasi, Rageshree | Adana, Bhimpalasi, Rageshree, suddha kalyan, Komal Bhairav |
| BIHU_012 | Laje Moru Moru | Bhupen Hazarika | Bihu | Bhoopali | 0.9350 | Puriya | 0.9348 | 0.0002 | Bhoopali, Puriya, Darbari | Bhoopali, Puriya, Darbari, Asavari, Adana |
| BIHU_013 | Ki Sai Aso | Bhupen Hazarika | Bihu | Ahir Bhairav | 0.9378 | Asavari | 0.9377 | 0.0001 | Ahir Bhairav, Asavari, Adana | Ahir Bhairav, Asavari, Adana, Bhoopali, Puriya |
| BIHU_014 | Bihureno Birina | Bhupen Hazarika | Bihu | Marva | 0.9399 | Bahar | 0.9386 | 0.0013 | Marva, Bahar, Khamaj | Marva, Bahar, Khamaj, Rageshree, Ahir Bhairav |
| BIHU_015 | Duetar Boraghor | Bhupen Hazarika | Bihu | Yaman Kalyan | 0.9327 | Asavari | 0.9314 | 0.0013 | Yaman Kalyan, Asavari, Jhinjhoti | Yaman Kalyan, Asavari, Jhinjhoti, Darbari, Bhoopali |
| BIHU_016 | Sangor Chira Pitha | Bhupen Hazarika | Bihu | Adana | 0.9341 | Yaman or Kalyan | 0.9337 | 0.0004 | Adana, Yaman or Kalyan, Bhoopali | Adana, Yaman or Kalyan, Bhoopali, Puriya, Asavari |
| BIHU_017 | Kopou Phul Phulile | Khagen Mahanta | Bihu | Bhoopali | 0.9308 | Puriya | 0.9258 | 0.0050 | Bhoopali, Puriya, Barwa | Bhoopali, Puriya, Barwa, Megh, Asavari |
| BIHU_018 | Amona Dhanoke | Khagen Mahanta | Bihu | Desh | 0.9321 | Megh | 0.9310 | 0.0012 | Desh, Megh, Khamaj | Desh, Megh, Khamaj, Asavari, Bhupal Todi |
| BIHU_019 | O Dhei Oi | Khagen Mahanta | Bihu | Megh | 0.9416 | Bhairavi | 0.9335 | 0.0081 | Megh, Bhairavi, Desh | Megh, Bhairavi, Desh, Bahar, Bageshree |
| BIHU_020 | Dhansiri Pare Hoi | Khagen Mahanta | Bihu | Bhairavi | 0.9316 | Ahir Bhairav | 0.9304 | 0.0012 | Bhairavi, Ahir Bhairav, Shree | Bhairavi, Ahir Bhairav, Shree, Bhatiyar, Lalit |
| BIHU_021 | Petore Juruni Bhat | Khagen Mahanta | Bihu | Megh Malhar | 0.9354 | Malhar | 0.9354 | 0.0000 | Megh Malhar, Malhar, Adana | Megh Malhar, Malhar, Adana, Bilaval, Bhairavi |
| BIHU_022 | Aag Fale Dekhilo | Khagen Mahanta | Bihu | Bahar | 0.9400 | Deepak | 0.9375 | 0.0025 | Bahar, Deepak, Bhairavi | Bahar, Deepak, Bhairavi, Rageshree, Bilaval |
| BIHU_023 | Aghunor Potharot | Khagen Mahanta | Bihu | Bhoopali | 0.9258 | Sorath | 0.9236 | 0.0022 | Bhoopali, Sorath, sarang | Bhoopali, Sorath, sarang, Darbari, Desh |
| BIHU_024 | Ebela Hale Bau | Khagen Mahanta | Bihu | Bhoopali | 0.9140 | Abhogi | 0.9121 | 0.0019 | Bhoopali, Abhogi, Megh | Bhoopali, Abhogi, Megh, Yaman or Kalyan, Darbari |
| BIHU_025 | Senai o Tumake Bisari | Khagen Mahanta | Bihu | Ahir Bhairav | 0.9319 | Megh | 0.9283 | 0.0036 | Ahir Bhairav, Megh, Bahar | Ahir Bhairav, Megh, Bahar, Bilaval, Rageshree |
| BIHU_026 | Bohag Ahil Buli | Shanta Uzir | Bihu | Ahir Bhairav | 0.9448 | Bhoopali | 0.9410 | 0.0038 | Ahir Bhairav, Bhoopali, Sohni | Ahir Bhairav, Bhoopali, Sohni, Adana, Bhimpalasi |
| BIHU_027 | Tumi Nasoni | Shanta Uzir | Bihu | Ahir Bhairav | 0.9331 | Yaman Kalyan | 0.9299 | 0.0032 | Ahir Bhairav, Yaman Kalyan, Darbari | Ahir Bhairav, Yaman Kalyan, Darbari, Sohni, Abhogi |
| BIHU_028 | Xunjone Nuxudhiba | Zubeen Garg, Vitali Das | Bihu | Bhoopali | 0.9347 | Bahar | 0.9335 | 0.0012 | Bhoopali, Bahar, Ahir Bhairav | Bhoopali, Bahar, Ahir Bhairav, Madhuvanti, Malkauns |
| BIHU_029 | Morone Phulonit | Zubeen Garg, Vitali Das | Bihu | Bhatiyar | 0.9391 | Darbari | 0.9380 | 0.0011 | Bhatiyar, Darbari, Alhaiya Bilaval | Bhatiyar, Darbari, Alhaiya Bilaval, Ahir Bhairav, Adana |
| BIHU_030 | Nila Nila Akhore | Zubeen Garg, Nirmali Das | Bihu | Bilaskhani Todi | 0.9336 | Bahar | 0.9333 | 0.0004 | Bilaskhani Todi, Bahar, Adana | Bilaskhani Todi, Bahar, Adana, Puriya, Bhoopali |
| BIHU_031 | Kahili puwate | Zubeen Garg, Priyanka Bharali | Bihu | Malkauns | 0.9409 | Darbari | 0.9390 | 0.0019 | Malkauns, Darbari, suddha kalyan | Malkauns, Darbari, suddha kalyan, Hindol, Ahir Bhairav |
| BIHU_032 | Rohare O Rohedoi | Bhitali Das | Bihu | Sohni | 0.9367 | Khamaj | 0.9349 | 0.0018 | Sohni, Khamaj, Bilaval | Sohni, Khamaj, Bilaval, Abhogi, Darbari |
| BIHU_033 | Dhengkoli | Navanita Sharma | Bihu | Khamaj | 0.9378 | Asavari | 0.9355 | 0.0023 | Khamaj, Asavari, Madhuvanti | Khamaj, Asavari, Madhuvanti, Yaman Kalyan, Bhimpalasi |
| BIHU_034 | Gabhoru Bihu | Deeplina Deka | Bihu | Khamaj | 0.9449 | Bhairav | 0.9412 | 0.0037 | Khamaj, Bhairav, Bihag | Khamaj, Bhairav, Bihag, Darbari, Ahir Bhairav |
| BIHU_035 | Bohagi | Rupankrita and Alankrita | Bihu | Bahar | 0.9335 | Adana | 0.9314 | 0.0021 | Bahar, Adana, Megh | Bahar, Adana, Megh, Khamaj, Kedar |
| BIHU_036 | Podulit Bihu | Rupankrita and Alankrita | Bihu | Bageshree | 0.9383 | Desh | 0.9361 | 0.0023 | Bageshree, Desh, Khamaj | Bageshree, Desh, Khamaj, Darbari, Puriya |
| BIHU_037 | Tok Dekhi Mor Gaa | Abhishruti Bezbaruah | Bihu | Madhuvanti | 0.9381 | Alhaiya Bilaval | 0.9371 | 0.0009 | Madhuvanti, Alhaiya Bilaval, Megh | Madhuvanti, Alhaiya Bilaval, Megh, Puriya, Jaunpuri |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dithi Das | Dhamail | Darbari | 0.9274 | Madhuvanti | 0.9255 | 0.0019 | Darbari, Madhuvanti, Poorvi | Darbari, Madhuvanti, Poorvi, Miyan Ki Todi orTodi or Darbari Todi |
| DHAMAIL_002 | আর যাবো না জলে | Dithi Das | Dhamail | Khamaj | 0.9325 | Madhuvanti | 0.9313 | 0.0012 | Khamaj, Madhuvanti, Poorvi | Khamaj, Madhuvanti, Poorvi, Darbari, Jaunpuri |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dithi Das | Dhamail | Adana | 0.9343 | Khamaj | 0.9335 | 0.0009 | Adana, Khamaj, Kedar | Adana, Khamaj, Kedar, Lalit, Madhuvanti |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dithi Das | Dhamail | Lalit | 0.9354 | Bhatiyar | 0.9345 | 0.0008 | Lalit, Bhatiyar, Bhairav | Lalit, Bhatiyar, Bhairav, Nat Bhairav, Miyan Ki Todi orTodi or Darbari Todi |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dithi Das | Dhamail | Alhaiya Bilaval | 0.9257 | Nat Bhairav | 0.9242 | 0.0014 | Alhaiya Bilaval, Nat Bhairav, Darbari | Alhaiya Bilaval, Nat Bhairav, Darbari, Miyan Ki Todi orTodi or Darbari Todi, Bageshree |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Uma Das | Dhamail | Purba | 0.9376 | Malkauns | 0.9354 | 0.0021 | Purba, Malkauns, Megh | Purba, Malkauns, Megh, sarang, Yaman Kalyan |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Uma Das | Dhamail | Abhogi | 0.9279 | sarang | 0.9258 | 0.0020 | Abhogi, sarang, Jaunpuri | Abhogi, sarang, Jaunpuri, Desh, Komal Bhairav |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Uma Das | Dhamail | Bahar | 0.9486 | Miyan Ki Todi orTodi or Darbari Todi | 0.9476 | 0.0010 | Bahar, Miyan Ki Todi orTodi or Darbari Todi, Purba | Bahar, Miyan Ki Todi orTodi or Darbari Todi, Purba, Bilaval, Darbari |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Uma Das | Dhamail | Basant Mukhari | 0.9399 | Malkauns | 0.9392 | 0.0007 | Basant Mukhari, Malkauns, sarang | Basant Mukhari, Malkauns, sarang, Bhairav Bahar, Miyan Ki Todi orTodi or Darbari Todi |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Uma Das | Dhamail | Jaunpuri | 0.9390 | Purba | 0.9356 | 0.0035 | Jaunpuri, Purba, Gujari Todi | Jaunpuri, Purba, Gujari Todi, Megh, sarang |
| DHAMAIL_011 | বল গো বল সখি | Habib Simu/Shimu | Dhamail | Marva | 0.9381 | Sorath | 0.9374 | 0.0006 | Marva, Sorath, Bhimpalasi | Marva, Sorath, Bhimpalasi, Darbari, Desh |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Habib Simu/Shimu | Dhamail | Darbari | 0.9444 | Multani | 0.9407 | 0.0038 | Darbari, Multani, Purba | Darbari, Multani, Purba, Sorath, Madhuvanti |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Habib Simu/Shimu | Dhamail | Darbari | 0.9474 | Bageshree | 0.9434 | 0.0040 | Darbari, Bageshree, Asavari | Darbari, Bageshree, Asavari, Khamaj |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Habib Simu/Shimu | Dhamail | Purba | 0.9258 | Alhaiya Bilaval | 0.9233 | 0.0025 | Purba, Alhaiya Bilaval, Madhuvanti | Purba, Alhaiya Bilaval, Madhuvanti, Adana, Bhairavi |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Habib Simu/Shimu | Dhamail | Madhuvanti | 0.9452 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | 0.0070 | Madhuvanti, Miyan Ki Todi orTodi or Darbari Todi, Multani | Madhuvanti, Miyan Ki Todi orTodi or Darbari Todi, Multani, Nat Bhairav, Asavari |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Jumiar Shanta | Dhamail | Bihag | 0.9409 | Kedar | 0.9408 | 0.0001 | Bihag, Kedar, Multani | Bihag, Kedar, Multani, Bhimpalasi, Miyan Ki Todi orTodi or Darbari Todi |
| DHAMAIL_017 | বলো গো বলো সই | Pausali Banerjee | Dhamail | Khamaj | 0.9351 | Bhoopali | 0.9330 | 0.0021 | Khamaj, Bhoopali, Yaman Kalyan | Khamaj, Bhoopali, Yaman Kalyan, Ahir Bhairav, Malkauns |
| DHAMAIL_018 | কত আদরে আদরে | Pausali Banerjee | Dhamail | Bhupal Todi | 0.9340 | Desh | 0.9286 | 0.0053 | Bhupal Todi, Desh, Yaman Kalyan | Bhupal Todi, Desh, Yaman Kalyan, Barwa, Poorvi |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Pausali Banerjee | Dhamail | Darbari | 0.9376 | Asavari | 0.9334 | 0.0042 | Darbari, Asavari, Adana | Darbari, Asavari, Adana |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Pausali Banerjee | Dhamail | Darbari | 0.9275 | Asavari | 0.9245 | 0.0030 | Darbari, Asavari, Sorath | Darbari, Asavari, Sorath, Khamaj, Bageshree |
| DHAMAIL_021 | জলের ঘাটে | Pausali Banerjee | Dhamail | Bihag | 0.9414 | Yaman Kalyan | 0.9299 | 0.0115 | Bihag, Yaman Kalyan, Kedar | Bihag, Yaman Kalyan, Kedar, Nat Bhairav, Darbari |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Maadal | Dhamail | Poorvi | 0.9384 | Darbari | 0.9369 | 0.0015 | Poorvi, Darbari, Lalit | Poorvi, Darbari, Lalit, Desh, Bahar |
| DHAMAIL_023 | Tara Lowo Go Jol | Maadal | Dhamail | Khamaj | 0.9349 | Bilaskhani Todi | 0.9298 | 0.0052 | Khamaj, Bilaskhani Todi, Bilaval | Khamaj, Bilaskhani Todi, Bilaval, Kedar, Darbari |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Maadal | Dhamail | Desi | 0.9446 | Megh | 0.9420 | 0.0026 | Desi, Megh, Pahadi | Desi, Megh, Pahadi, Nat Bhairav, Ahir Bhairav |
| DHAMAIL_025 | Nrittangon Sunamganj | Debdas Chowdhury | Dhamail | Bhairavi | 0.9243 | Malkauns | 0.9217 | 0.0025 | Bhairavi, Malkauns, Madhuvanti | Bhairavi, Malkauns, Madhuvanti, Sorath, Rageshree |
| DHAMAIL_026 | জলার জলার ঘাটে | Himel Dev Rahul | Dhamail | Marva | 0.9157 | Ahir Bhairav | 0.9156 | 0.0002 | Marva, Ahir Bhairav, Gauri | Marva, Ahir Bhairav, Gauri, Rageshree, suddha kalyan |
| DHAMAIL_027 | অচেনা এক রূপবতী | Samim Siddiki | Dhamail | Ahir Bhairav | 0.9444 | Adana | 0.9435 | 0.0009 | Ahir Bhairav, Adana, Megh | Ahir Bhairav, Adana, Megh |
| DHAMAIL_028 | কুঞ্জের মাঝে | Madol Folk Band | Dhamail | Megh | 0.9370 | Sorath | 0.9343 | 0.0027 | Megh, Sorath, Yaman or Kalyan | Megh, Sorath, Yaman or Kalyan |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Suman Dewan | Dhamail | Adana | 0.9412 | Sorath | 0.9384 | 0.0028 | Adana, Sorath, Pahadi | Adana, Sorath, Pahadi, Malkauns |
| DHAMAIL_030 | জানো নি গো সই | Jabel Raza | Dhamail | Asavari | 0.9140 | Puriya | 0.9119 | 0.0021 | Asavari, Puriya, Adana | Asavari, Puriya, Adana, Sorath, Alhaiya Bilaval |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Diwaliben Bhil | Garba | Malkauns | 0.9390 | Darbari | 0.9384 | 0.0006 | Malkauns, Darbari, Sorath | Malkauns, Darbari, Sorath, Asavari, Bhairavi |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Vibha Desai, Sharavan Madhurai | Garba | Ahir Bhairav | 0.9412 | Bilaskhani Todi | 0.9394 | 0.0019 | Ahir Bhairav, Bilaskhani Todi, Basant Mukhari | Ahir Bhairav, Bilaskhani Todi, Basant Mukhari, Bhupal Todi, Bhairav Bahar |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Geeta Rabari | Garba | Alhaiya Bilaval | 0.9447 | Lalit | 0.9437 | 0.0010 | Alhaiya Bilaval, Lalit, Darbari | Alhaiya Bilaval, Lalit, Darbari |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Lalita Ghodadra | Garba | Megh | 0.9466 | suddha kalyan | 0.9457 | 0.0009 | Megh, suddha kalyan, Desh | Megh, suddha kalyan, Desh, Sorath, Darbari |
| GARBA_005 | Amba Aavo To Ramiye | Lalita Ghodadra | Garba | Adana | 0.9489 | Yaman or Kalyan | 0.9442 | 0.0047 | Adana, Yaman or Kalyan, Megh | Adana, Yaman or Kalyan, Megh |
| GARBA_006 | Sonal Garbo Shire | Kishor Manraja | Garba | Megh Malhar | 0.9469 | Komal Bhairav | 0.9466 | 0.0003 | Megh Malhar, Komal Bhairav, Gujari Todi | Megh Malhar, Komal Bhairav, Gujari Todi, Miyan Ki Todi orTodi or Darbari Todi, Malkauns |
| GARBA_007 | Maa No Garbo Re | Geeta Rabari | Garba | Asavari | 0.9410 | Darbari | 0.9401 | 0.0009 | Asavari, Darbari, Alhaiya Bilaval | Asavari, Darbari, Alhaiya Bilaval, Madhuvanti, Adana |
| GARBA_008 | Unchi Talavdi Ni Kor | Asha Bhosle | Garba | Asavari | 0.9397 | Marva | 0.9344 | 0.0053 | Asavari, Marva, Kafi | Asavari, Marva, Kafi, Bahar, Hemant |
| GARBA_009 | Aavya Maa Na Norta | Pamela Jain, Divya Kumar | Garba | Adana | 0.9572 | Marva | 0.9544 | 0.0027 | Adana, Marva, Rageshree | Adana, Marva, Rageshree, Bahar, Desh |
| GARBA_010 | Khodiyar Maa Nu Holdu | Kinjal Dave | Garba | Alhaiya Bilaval | 0.9336 | Bhimpalasi | 0.9319 | 0.0017 | Alhaiya Bilaval, Bhimpalasi, Darbari | Alhaiya Bilaval, Bhimpalasi, Darbari, Marva, Lalit |
| GARBA_011 | Sathiya Puravo | Damyanti Bardai | Garba | Megh | 0.9418 | Barwa | 0.9401 | 0.0017 | Megh, Barwa, Puriya | Megh, Barwa, Puriya, Yaman Kalyan |
| GARBA_012 | Amba Aave Aarasur Vadi | Meena Patel | Garba | Yaman or Kalyan | 0.9249 | Khamaj | 0.9193 | 0.0055 | Yaman or Kalyan, Khamaj, Bihag | Yaman or Kalyan, Khamaj, Bihag, Puriya, Megh |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Meena Patel, Praful Dave | Garba | Megh | 0.9419 | Puriya | 0.9411 | 0.0008 | Megh, Puriya, Bilaskhani Todi | Megh, Puriya, Bilaskhani Todi, Bhairav Bahar, Gujari Todi |
| GARBA_014 | Garba No Rang Lagiyo | Meena Patel, Praful Dave | Garba | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | Durga | 0.9379 | 0.0003 | Miyan Ki Todi orTodi or Darbari Todi, Durga, Adana | Miyan Ki Todi orTodi or Darbari Todi, Durga, Adana, Madhuvanti, Jaunpuri |
| GARBA_015 | Hindole Zule Amba | Meena Patel | Garba | Bhoopali | 0.9291 | Barwa | 0.9284 | 0.0007 | Bhoopali, Barwa, Darbari | Bhoopali, Barwa, Darbari, Bilaval, suddha kalyan |
| GARBA_016 | Jay Ambe Jagdambe Maa | Mital Gadhvi | Garba | Adana | 0.9402 | Bhoopali | 0.9379 | 0.0023 | Adana, Bhoopali, Megh | Adana, Bhoopali, Megh, Bhairavi, Pahadi |
| GARBA_017 | Halaji Tara Hath Vakhanu | Aditya Gadhvi | Garba | Darbari | 0.9499 | Asavari | 0.9451 | 0.0048 | Darbari, Asavari, Adana | Darbari, Asavari, Adana, Pahadi |
| GARBA_018 | Dholida Dhol Re Vagad | Farida Mir | Garba | Bhoopali | 0.9344 | Bhimpalasi | 0.9337 | 0.0007 | Bhoopali, Bhimpalasi, Asavari | Bhoopali, Bhimpalasi, Asavari, Puriya, sarang |
| GARBA_019 | Pankhida O Pankhida | Hemant Chauhan | Garba | Jaunpuri | 0.9259 | Lalit | 0.9243 | 0.0016 | Jaunpuri, Lalit, Desh | Jaunpuri, Lalit, Desh, Durga, Adana |
| GARBA_020 | Tu To Kali Ne Kalyani | Hemant Chauhan | Garba | Barwa | 0.9305 | Bhairavi | 0.9281 | 0.0024 | Barwa, Bhairavi, Yaman Kalyan | Barwa, Bhairavi, Yaman Kalyan, Bihag, Purba |
| GARBA_021 | He Jagjanani | Hemant Chauhan | Garba | Kafi | 0.9288 | Desh | 0.9251 | 0.0038 | Kafi, Desh, Yaman Kalyan | Kafi, Desh, Yaman Kalyan, Alhaiya Bilaval, Sorath |
| GARBA_022 | Chotile Dakla Vagya | Hemant Chauhan | Garba | Desh | 0.9191 | Basant Mukhari | 0.9151 | 0.0040 | Desh, Basant Mukhari, Megh Malhar | Desh, Basant Mukhari, Megh Malhar, Malhar, Durga |
| GARBA_023 | Jay Aadhyashakti | Hemant Chauhan | Garba | Purba | 0.9348 | nan | 0.0000 | 0.9348 | Purba, nan, nan | Purba |
| GARBA_024 | Dam Dam Dakla Vagya | Hemant Chauhan | Garba | Bahar | 0.9234 | Gauri | 0.9228 | 0.0006 | Bahar, Gauri, Basant Mukhari | Bahar, Gauri, Basant Mukhari, Khamaj, Puriya |
| GARBA_025 | Sonala Vatakadi | Praful Dave, Damyanti Bardai | Garba | Marva | 0.9513 | Bilaval | 0.9485 | 0.0028 | Marva, Bilaval, Kedar | Marva, Bilaval, Kedar, Bhairav, Miyan Ki Todi orTodi or Darbari Todi |
| GARBA_026 | Kaho Poonam Na Chand | Shailendra Bharti | Garba | Adana | 0.9549 | Alhaiya Bilaval | 0.9547 | 0.0003 | Adana, Alhaiya Bilaval, Asavari | Adana, Alhaiya Bilaval, Asavari, Malkauns |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Praful Dave, Meena Patel | Garba | Basant | 0.9439 | Desh | 0.9421 | 0.0018 | Basant, Desh, Jaunpuri | Basant, Desh, Jaunpuri, Madhuvanti, Bhinna Shadja |
| GARBA_028 | Lejo Rasiya Re Rumal | Arvind Barot, Vatsala Patil | Garba | Marva | 0.9475 | Asavari | 0.9436 | 0.0039 | Marva, Asavari, Sorath | Marva, Asavari, Sorath, Jaunpuri, Bhimpalasi |
| GARBA_029 | Pavana Dungre | Praful Dave | Garba | Darbari | 0.9384 | Desi | 0.9379 | 0.0005 | Darbari, Desi, Bihag | Darbari, Desi, Bihag, Kafi, Miyan Ki Todi orTodi or Darbari Todi |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Arvind Barot | Garba | Khamaj | 0.9388 | Pahadi | 0.9380 | 0.0008 | Khamaj, Pahadi, Gauri | Khamaj, Pahadi, Gauri, Miyan Ki Todi orTodi or Darbari Todi, Sorath |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Arvind Barot, Bhavna Rana | Garba | Nat Bhairav | 0.9456 | Pahadi | 0.9453 | 0.0002 | Nat Bhairav, Pahadi, Purba | Nat Bhairav, Pahadi, Purba, Bihag, Khamaj |
| GHOOMAR_001 | Ghoomar | Seema Mishra | Ghoomar | Khamaj | 0.9272 | Megh | 0.9263 | 0.0009 | Khamaj, Megh, Malkauns | Khamaj, Megh, Malkauns, Basant Mukhari, Bibhas |
| GHOOMAR_002 | Gangore/Gangaur | Seema Mishra | Ghoomar | Bhairav | 0.9523 | Miyan Ki Todi orTodi or Darbari Todi | 0.9458 | 0.0064 | Bhairav, Miyan Ki Todi orTodi or Darbari Todi, Adana | Bhairav, Miyan Ki Todi orTodi or Darbari Todi, Adana |
| GHOOMAR_003 | Choumaso | Seema Mishra | Ghoomar | Yaman or Kalyan | 0.9322 | Bahar | 0.9316 | 0.0006 | Yaman or Kalyan, Bahar, suddha kalyan | Yaman or Kalyan, Bahar, suddha kalyan, Purba, Bhoopali |
| GHOOMAR_004 | More Bole Re | Seema Mishra | Ghoomar | Bhairav | 0.9425 | Adana | 0.9379 | 0.0045 | Bhairav, Adana, Malkauns | Bhairav, Adana, Malkauns, Basant Mukhari, Khamaj |
| GHOOMAR_005 | Podina | Seema Mishra | Ghoomar | Asavari | 0.9418 | Malkauns | 0.9388 | 0.0030 | Asavari, Malkauns, Bhairav | Asavari, Malkauns, Bhairav, Purba, Marva |
| GHOOMAR_006 | Talariya Magriya | Seema Mishra | Ghoomar | Adana | 0.9249 | Khamaj | 0.9217 | 0.0032 | Adana, Khamaj, Bhairav | Adana, Khamaj, Bhairav, Marva, Bhoopali |
| GHOOMAR_007 | Panihari Ji | Seema Mishra | Ghoomar | Bhoopali | 0.9394 | Yaman or Kalyan | 0.9348 | 0.0046 | Bhoopali, Yaman or Kalyan, Megh | Bhoopali, Yaman or Kalyan, Megh, Adana |
| GHOOMAR_008 | Roomal/Rumal | Seema Mishra | Ghoomar | Bilaval | 0.9330 | Yaman Kalyan | 0.9327 | 0.0004 | Bilaval, Yaman Kalyan, Darbari | Bilaval, Yaman Kalyan, Darbari, Khamaj, Sohni |
| GHOOMAR_009 | Gorband | Seema Mishra | Ghoomar | Bhairav | 0.9341 | Asavari | 0.9305 | 0.0036 | Bhairav, Asavari, suddha kalyan | Bhairav, Asavari, suddha kalyan, Malkauns, Ahir Bhairav |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Seema Mishra | Ghoomar | Bhairav | 0.9440 | Asavari | 0.9402 | 0.0039 | Bhairav, Asavari, Megh | Bhairav, Asavari, Megh, Yaman or Kalyan, Malkauns |
| GHOOMAR_011 | Hivde Ro Haar | Seema Mishra | Ghoomar | Barwa | 0.9375 | Khamaj | 0.9353 | 0.0021 | Barwa, Khamaj, Ahir Bhairav | Barwa, Khamaj, Ahir Bhairav, Darbari, Asavari |
| GHOOMAR_012 | Chirmi | Seema Mishra | Ghoomar | Yaman Kalyan | 0.9401 | Yaman or Kalyan | 0.9378 | 0.0022 | Yaman Kalyan, Yaman or Kalyan, Bhoopali | Yaman Kalyan, Yaman or Kalyan, Bhoopali, Adana, Pahadi |
| GHOOMAR_013 | Hichki (Aave Hichki) | Seema Mishra | Ghoomar | Darbari | 0.9553 | Basant | 0.9499 | 0.0054 | Darbari, Basant, Bahar | Darbari, Basant, Bahar, Adana, Bibhas |
| GHOOMAR_014 | Dal Badal | Seema Mishra | Ghoomar | Adana | 0.9427 | Nat Bhairav | 0.9402 | 0.0025 | Adana, Nat Bhairav, Miyan Ki Todi orTodi or Darbari Todi | Adana, Nat Bhairav, Miyan Ki Todi orTodi or Darbari Todi, Megh, Yaman or Kalyan |
| GHOOMAR_015 | Samdariyo | Seema Mishra | Ghoomar | Miyan Ki Todi orTodi or Darbari Todi | 0.9456 | Bahar | 0.9454 | 0.0002 | Miyan Ki Todi orTodi or Darbari Todi, Bahar, Durga | Miyan Ki Todi orTodi or Darbari Todi, Bahar, Durga, Madhuvanti, Megh |
| GHOOMAR_016 | Jeero | Seema Mishra | Ghoomar | Asavari | 0.9399 | Yaman or Kalyan | 0.9396 | 0.0003 | Asavari, Yaman or Kalyan, Adana | Asavari, Yaman or Kalyan, Adana, sarang |
| GHOOMAR_017 | Loonga Ro Batawo | Seema Mishra | Ghoomar | Darbari | 0.9259 | Bhimpalasi | 0.9227 | 0.0032 | Darbari, Bhimpalasi, Desi | Darbari, Bhimpalasi, Desi, Basant Mukhari, Megh |
| GHOOMAR_018 | Taran Ri Chunadi | Seema Mishra | Ghoomar | Adana | 0.9214 | Megh | 0.9207 | 0.0008 | Adana, Megh, Jaunpuri | Adana, Megh, Jaunpuri, Pahadi |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Seema Mishra | Ghoomar | Alhaiya Bilaval | 0.9360 | suddha kalyan | 0.9205 | 0.0156 | Alhaiya Bilaval, suddha kalyan, Pahadi | Alhaiya Bilaval, suddha kalyan, Pahadi, Bhimpalasi, Adana |
| GHOOMAR_020 | Kangasiyo | Seema Mishra | Ghoomar | Ahir Bhairav | 0.9296 | Desh | 0.9234 | 0.0062 | Ahir Bhairav, Desh, Bageshree | Ahir Bhairav, Desh, Bageshree, Bhairav, Miyan Ki Todi orTodi or Darbari Todi |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Seema Mishra | Ghoomar | Bhairav | 0.9455 | Ahir Bhairav | 0.9445 | 0.0010 | Bhairav, Ahir Bhairav, Desh | Bhairav, Ahir Bhairav, Desh, suddha kalyan, Miyan Ki Todi orTodi or Darbari Todi |
| GHOOMAR_022 | Kesariya Ro Pag | Seema Mishra | Ghoomar | Asavari | 0.9419 | Khamaj | 0.9373 | 0.0046 | Asavari, Khamaj, Sohni | Asavari, Khamaj, Sohni, Kedar, Bhoopali |
| GHOOMAR_023 | Naina Ra Lobhi | Seema Mishra | Ghoomar | Miyan Ki Todi orTodi or Darbari Todi | 0.9314 | Pahadi | 0.9305 | 0.0009 | Miyan Ki Todi orTodi or Darbari Todi, Pahadi, Multani | Miyan Ki Todi orTodi or Darbari Todi, Pahadi, Multani, Rageshree, Madhuvanti |
| GHOOMAR_024 | Bajudan Ri Loom | Seema Mishra | Ghoomar | Adana | 0.9280 | Khamaj | 0.9270 | 0.0010 | Adana, Khamaj, Miyan Ki Todi orTodi or Darbari Todi | Adana, Khamaj, Miyan Ki Todi orTodi or Darbari Todi, Megh Malhar, Barwa |
| GHOOMAR_025 | Lehariyo | Seema Mishra | Ghoomar | Darbari | 0.9203 | Bhairav | 0.9193 | 0.0010 | Darbari, Bhairav, Kedar | Darbari, Bhairav, Kedar, Khamaj, Yaman Kalyan |
| GHOOMAR_026 | Ital Peetal | Seema Mishra | Ghoomar | Durga | 0.9281 | Pahadi | 0.9254 | 0.0027 | Durga, Pahadi, Darbari | Durga, Pahadi, Darbari, Alhaiya Bilaval, Miyan Ki Todi orTodi or Darbari Todi |
| GHOOMAR_027 | Moriya | Seema Mishra | Ghoomar | Bhatiyar | 0.9387 | Megh | 0.9352 | 0.0036 | Bhatiyar, Megh, Lalit | Bhatiyar, Megh, Lalit, Pahadi, Madhuvanti |
| GHOOMAR_028 | Cham Cham Chake Chundari | Seema Mishra | Ghoomar | Gujari Todi | 0.9231 | Madhuvanti | 0.9223 | 0.0007 | Gujari Todi, Madhuvanti, Desh | Gujari Todi, Madhuvanti, Desh, Multani, Adana |
| GHOOMAR_029 | Hathelyan Re Beech | Seema Mishra | Ghoomar | Barwa | 0.9361 | Khamaj | 0.9337 | 0.0023 | Barwa, Khamaj, Megh | Barwa, Khamaj, Megh, Bahar, Desh |
| GHOOMAR_030 | Dar Lanjo | Seema Mishra | Ghoomar | Bhairav | 0.9436 | Asavari | 0.9419 | 0.0017 | Bhairav, Asavari, Bhoopali | Bhairav, Asavari, Bhoopali, Adana, Megh Malhar |
| JHUMUR_001 | Kati Khane Bole | Dulal Manki | Jhumur | Marva | 0.9283 | Bhatiyar | 0.9261 | 0.0022 | Marva, Bhatiyar, Madhuvanti | Marva, Bhatiyar, Madhuvanti, Bhinna Shadja, Malkauns |
| JHUMUR_0010 | Ekadoshi Ratiya | Zubeen Garg | Jhumur | Lalit | 0.9387 | Bilaval | 0.9379 | 0.0008 | Lalit, Bilaval, Megh | Lalit, Bilaval, Megh, Barwa, Madhuvanti |
| JHUMUR_0011 | Barire Simare | Zubeen Garg | Jhumur | Ahir Bhairav | 0.9385 | Darbari | 0.9343 | 0.0041 | Ahir Bhairav, Darbari, Pahadi | Ahir Bhairav, Darbari, Pahadi, suddha kalyan |
| JHUMUR_0012 | Mor Dhoni Go | Zubeen Garg | Jhumur | Malkauns | 0.9293 | Madhuvanti | 0.9192 | 0.0101 | Malkauns, Madhuvanti, Bilaval | Malkauns, Madhuvanti, Bilaval, Asavari, Sorath |
| JHUMUR_0013 | Jhuli Jhule | Zubeen Garg | Jhumur | Desi | 0.9251 | Madhuvanti | 0.9251 | 0.0000 | Desi, Madhuvanti, Khamaj | Desi, Madhuvanti, Khamaj, Marva, Ahir Bhairav |
| JHUMUR_0014 | Kansiya | Amrit Tanti | Jhumur | Bageshree | 0.9427 | Kedar | 0.9418 | 0.0009 | Bageshree, Kedar, Asavari | Bageshree, Kedar, Asavari, Yaman Kalyan, Darbari |
| JHUMUR_0015 | Jhatka | Amrit Tanti | Jhumur | Bhoopali | 0.9381 | Sorath | 0.9366 | 0.0015 | Bhoopali, Sorath, Megh | Bhoopali, Sorath, Megh, Barwa, Adana |
| JHUMUR_0016 | Gohona | Amrit Tanti | Jhumur | Bageshree | 0.9419 | Khamaj | 0.9387 | 0.0032 | Bageshree, Khamaj, Darbari | Bageshree, Khamaj, Darbari, Sohni, Asavari |
| JHUMUR_0017 | Hariya | Amrit Tanti | Jhumur | Kedar | 0.9342 | Darbari | 0.9303 | 0.0039 | Kedar, Darbari, Bhoopali | Kedar, Darbari, Bhoopali, Asavari |
| JHUMUR_0018 | Koromoti | Amrit Tanti | Jhumur | Khamaj | 0.9395 | Bageshree | 0.9390 | 0.0005 | Khamaj, Bageshree, Kedar | Khamaj, Bageshree, Kedar, Adana, Bilaval |
| JHUMUR_0019 | Sonamoni | Amrit Tanti | Jhumur | Adana | 0.9368 | Desh | 0.9367 | 0.0002 | Adana, Desh, Sorath | Adana, Desh, Sorath, Bahar, Barwa |
| JHUMUR_002 | Nistei | Dulal Manki | Jhumur | Bageshree | 0.9418 | Sohni | 0.9418 | 0.0000 | Bageshree, Sohni, Pahadi | Bageshree, Sohni, Pahadi, Marva, Sorath |
| JHUMUR_0020 | Chaliya Chhodi | Amrit Tanti | Jhumur | Bhoopali | 0.9277 | Malkauns | 0.9247 | 0.0030 | Bhoopali, Malkauns, Khamaj | Bhoopali, Malkauns, Khamaj |
| JHUMUR_0021 | Prem Nadi Tire Tire | Shaan Tanti | Jhumur | Bhairavi | 0.9313 | Ahir Bhairav | 0.9311 | 0.0002 | Bhairavi, Ahir Bhairav, Adana | Bhairavi, Ahir Bhairav, Adana, Sorath, Megh Malhar |
| JHUMUR_0022 | Deewana | Shaan Tanti | Jhumur | Marva | 0.9413 | Bhimpalasi | 0.9378 | 0.0035 | Marva, Bhimpalasi, Sorath | Marva, Bhimpalasi, Sorath, Desh, Adana |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Navin Tanti | Jhumur | Lalit | 0.9528 | Megh | 0.9527 | 0.0000 | Lalit, Megh, Adana | Lalit, Megh, Adana |
| JHUMUR_0024 | Laal Pari | Bejnath Tanti | Jhumur | Madhuvanti | 0.9358 | Sorath | 0.9358 | 0.0000 | Madhuvanti, Sorath, Desh | Madhuvanti, Sorath, Desh, Megh, Adana |
| JHUMUR_0025 | Tore Bina | Bejnath Tanti | Jhumur | Asavari | 0.9524 | Adana | 0.9521 | 0.0003 | Asavari, Adana, Bhoopali | Asavari, Adana, Bhoopali, Alhaiya Bilaval |
| JHUMUR_0026 | Koromake Akhora | Bejnath Tanti | Jhumur | Bilaval | 0.9437 | Darbari | 0.9414 | 0.0023 | Bilaval, Darbari, Lalit | Bilaval, Darbari, Lalit, Bhimpalasi, Khamaj |
| JHUMUR_0027 | The Jhumur Song | Neel Akash | Jhumur | Nat Bhairav | 0.9461 | Asavari | 0.9452 | 0.0009 | Nat Bhairav, Asavari, Bhairav | Nat Bhairav, Asavari, Bhairav, Adana, Megh |
| JHUMUR_0028 | Genda Phool | Neel Akash, Jyotika Baruha | Jhumur | Bageshree | 0.9461 | Kedar | 0.9430 | 0.0030 | Bageshree, Kedar, Adana | Bageshree, Kedar, Adana, Darbari, Bilaval |
| JHUMUR_0029 | Ja Ja Re Mini | Neel Akash | Jhumur | Ahir Bhairav | 0.9344 | Asavari | 0.9334 | 0.0010 | Ahir Bhairav, Asavari, Pahadi | Ahir Bhairav, Asavari, Pahadi, Miyan Ki Todi orTodi or Darbari Todi, Darbari |
| JHUMUR_003 | Kuilu Kuilu | Dulal Manki | Jhumur | Darbari | 0.9368 | Bhinna Shadja | 0.9347 | 0.0021 | Darbari, Bhinna Shadja, Bhairav | Darbari, Bhinna Shadja, Bhairav, Basant, Basant Mukhari |
| JHUMUR_0030 | Goya | Neel Akash | Jhumur | Adana | 0.9377 | Asavari | 0.9369 | 0.0008 | Adana, Asavari, nan | Adana, Asavari |
| JHUMUR_004 | Sajanigo | Dulal Manki | Jhumur | Desi | 0.9391 | Yaman or Kalyan | 0.9328 | 0.0063 | Desi, Yaman or Kalyan, Rageshree | Desi, Yaman or Kalyan, Rageshree, Adana, Basant |
| JHUMUR_005 | Sol Sokhi | Dulal Manki | Jhumur | Deepak | 0.9357 | Bhatiyar | 0.9334 | 0.0023 | Deepak, Bhatiyar, Marva | Deepak, Bhatiyar, Marva, Lalit, Bhairav |
| JHUMUR_006 | Aji Ei Suprabhate | Dulal Manki | Jhumur | Bilaval | 0.9240 | Bhairavi | 0.9224 | 0.0016 | Bilaval, Bhairavi, Kafi | Bilaval, Bhairavi, Kafi, Sorath, Marva |
| JHUMUR_007 | Kale Gori Muke | Dulal Manki | Jhumur | Nat Bhairav | 0.9320 | Malhar | 0.9309 | 0.0011 | Nat Bhairav, Malhar, Malkauns | Nat Bhairav, Malhar, Malkauns, Basant, Basant Mukhari |
| JHUMUR_008 | Tohe Gori Nawa | Dulal Manki | Jhumur | Bahar | 0.9298 | Komal Bhairav | 0.9280 | 0.0018 | Bahar, Komal Bhairav, Sohni | Bahar, Komal Bhairav, Sohni, Malkauns, Bhinna Shadja |
| JHUMUR_009 | Moi to Jabo Kam Jari | Zubeen Garg | Jhumur | Adana | 0.9310 | sarang | 0.9304 | 0.0007 | Adana, sarang, Miyan Ki Todi orTodi or Darbari Todi | Adana, sarang, Miyan Ki Todi orTodi or Darbari Todi, Darbari, Sohni |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Sulochana Chavan | Lavani | Desh | 0.9381 | Bilaskhani Todi | 0.9374 | 0.0007 | Desh, Bilaskhani Todi, Basant | Desh, Bilaskhani Todi, Basant, Puriya, Megh |
| LAVANI_002 | Padarawarti Jartaricha | Sulochana Chavan | Lavani | Marva | 0.9466 | Megh Malhar | 0.9454 | 0.0012 | Marva, Megh Malhar, Asavari | Marva, Megh Malhar, Asavari, Kedar, Nat Bhairav |
| LAVANI_003 | Solava Varis Dhokyacha | Sulochana Chavan | Lavani | Bahar | 0.9312 | Bhairavi | 0.9306 | 0.0006 | Bahar, Bhairavi, Adana | Bahar, Bhairavi, Adana, Komal Bhairav, Asavari |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Sulochana Chavan | Lavani | Bahar | 0.9427 | Bihag | 0.9410 | 0.0017 | Bahar, Bihag, Khamaj | Bahar, Bihag, Khamaj, Bageshree, Bhimpalasi |
| LAVANI_005 | Naav Gaav Kashala Pustay | Sulochana Chavan | Lavani | Nat Bhairav | 0.9527 | Rageshree | 0.9500 | 0.0027 | Nat Bhairav, Rageshree, Megh Malhar | Nat Bhairav, Rageshree, Megh Malhar, Malhar |
| LAVANI_006 | Kalidar Kapuri Paan | Sulochana Chavan | Lavani | Bahar | 0.9247 | Nat Bhairav | 0.9216 | 0.0031 | Bahar, Nat Bhairav, Kafi | Bahar, Nat Bhairav, Kafi, Basant, suddha kalyan |
| LAVANI_007 | Mala London Foreign Dakhwa | Sulochana Chavan | Lavani | Malkauns | 0.9369 | Bahar | 0.9315 | 0.0053 | Malkauns, Bahar, Yaman Kalyan | Malkauns, Bahar, Yaman Kalyan, Bilaval, Yaman or Kalyan |
| LAVANI_008 | Disla Ga Bai Disla | Usha Mangeshkar | Lavani | Kedar | 0.9498 | Asavari | 0.9493 | 0.0005 | Kedar, Asavari, Gauri | Kedar, Asavari, Gauri, Marva, Megh Malhar |
| LAVANI_009 | Chabidar Chabi | Usha Mangeshkar | Lavani | suddha kalyan | 0.9305 | Yaman or Kalyan | 0.9276 | 0.0029 | suddha kalyan, Yaman or Kalyan, Puriya | suddha kalyan, Yaman or Kalyan, Puriya, Pahadi, Desh |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Usha Mangeshkar | Lavani | Asavari | 0.9530 | Miyan Ki Todi orTodi or Darbari Todi | 0.9494 | 0.0036 | Asavari, Miyan Ki Todi orTodi or Darbari Todi, Marva | Asavari, Miyan Ki Todi orTodi or Darbari Todi, Marva, Bihag, Lalit |
| LAVANI_011 | Pikala Jambhul Todu Naka | Usha Mangeshkar | Lavani | Asavari | 0.9584 | Miyan Ki Todi orTodi or Darbari Todi | 0.9557 | 0.0026 | Asavari, Miyan Ki Todi orTodi or Darbari Todi, Bageshree | Asavari, Miyan Ki Todi orTodi or Darbari Todi, Bageshree, Rageshree, Puriya |
| LAVANI_012 | Naki Doli Chhaan | Usha Mangeshkar | Lavani | Puriya | 0.9578 | Kedar | 0.9494 | 0.0084 | Puriya, Kedar, Bageshree | Puriya, Kedar, Bageshree, Megh, Bihag |
| LAVANI_013 | Aai Majhya Lagnachi | Usha Mangeshkar | Lavani | Malhar | 0.9455 | Marva | 0.9439 | 0.0017 | Malhar, Marva, Bhoopali | Malhar, Marva, Bhoopali, Durga |
| LAVANI_014 | Wajle Ki Bara | Bela Shende | Lavani | Bahar | 0.9337 | Bhoopali | 0.9333 | 0.0004 | Bahar, Bhoopali, Sohni | Bahar, Bhoopali, Sohni, Puriya, Bhairavi |
| LAVANI_015 | Apsara Aali | Bela Shende | Lavani | Basant | 0.9305 | Sorath | 0.9297 | 0.0008 | Basant, Sorath, Bhimpalasi | Basant, Sorath, Bhimpalasi, Marva, Bhinna Shadja |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Bela Shende | Lavani | Shree | 0.9252 | Bahar | 0.9245 | 0.0007 | Shree, Bahar, Bhimpalasi | Shree, Bahar, Bhimpalasi, Sohni, Adana |
| LAVANI_017 | Kashyala Lavato | Bela Shende | Lavani | Barwa | 0.9223 | Darbari | 0.9137 | 0.0086 | Barwa, Darbari, Alhaiya Bilaval | Barwa, Darbari, Alhaiya Bilaval, Ahir Bhairav, Kafi |
| LAVANI_018 | Chhatishi Kavlun Dhara | Surekha Punekar | Lavani | Bahar | 0.9357 | Madhuvanti | 0.9350 | 0.0007 | Bahar, Madhuvanti, Puriya | Bahar, Madhuvanti, Puriya, Khamaj, Darbari |
| LAVANI_019 | Jhampar Majha Dheel | Surekha Punekar | Lavani | Yaman Kalyan | 0.9265 | Miyan Ki Todi orTodi or Darbari Todi | 0.9250 | 0.0015 | Yaman Kalyan, Miyan Ki Todi orTodi or Darbari Todi, Puriya | Yaman Kalyan, Miyan Ki Todi orTodi or Darbari Todi, Puriya, Malkauns, Desi |
| LAVANI_020 | Themba Thembana Kelai | Surekha Punekar | Lavani | suddha kalyan | 0.9305 | Yaman or Kalyan | 0.9276 | 0.0029 | suddha kalyan, Yaman or Kalyan, Puriya | suddha kalyan, Yaman or Kalyan, Puriya, Pahadi, Desh |
| LAVANI_021 | Sola Hazarat | Uttara Kelkar | Lavani | Marva | 0.9375 | Bilaskhani Todi | 0.9350 | 0.0025 | Marva, Bilaskhani Todi, Desh | Marva, Bilaskhani Todi, Desh, Bageshree, Puriya |
| LAVANI_022 | Hi Natrangi Naar | Ravindra Sathe | Lavani | Puriya | 0.9424 | Marva | 0.9405 | 0.0019 | Puriya, Marva, Bageshree | Puriya, Marva, Bageshree, Bhairav, Bhairavi |
| LAVANI_023 | Saaman Avjad Majha | Anand Shinde | Lavani | Yaman Kalyan | 0.9265 | Miyan Ki Todi orTodi or Darbari Todi | 0.9219 | 0.0046 | Yaman Kalyan, Miyan Ki Todi orTodi or Darbari Todi, Bageshree | Yaman Kalyan, Miyan Ki Todi orTodi or Darbari Todi, Bageshree, Malkauns, Desh |
| LAVANI_024 | Majha Taat Kar Aadhi | Anand Shinde | Lavani | suddha kalyan | 0.9290 | Yaman or Kalyan | 0.9283 | 0.0008 | suddha kalyan, Yaman or Kalyan, Puriya | suddha kalyan, Yaman or Kalyan, Puriya, Pahadi, Desh |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Asha Bhosle | Lavani | Yaman or Kalyan | 0.9356 | Yaman Kalyan | 0.9351 | 0.0006 | Yaman or Kalyan, Yaman Kalyan, Gauri | Yaman or Kalyan, Yaman Kalyan, Gauri, Durga, Bahar |
| LAVANI_026 | Bai Mazi Karangali Modali | Asha Bhosle | Lavani | Basant Mukhari | 0.9551 | Puriya | 0.9550 | 0.0001 | Basant Mukhari, Puriya, Bihag | Basant Mukhari, Puriya, Bihag, Rageshree, Darbari |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Pushpa Pagdhare | Lavani | Marva | 0.9522 | Bageshree | 0.9471 | 0.0051 | Marva, Bageshree, Asavari | Marva, Bageshree, Asavari, Purba, Bhoopali |
| LAVANI_028 | Navi Navi Navari Mee | Pushpa Pagdhare | Lavani | Malhar | 0.9410 | Khamaj | 0.9365 | 0.0045 | Malhar, Khamaj, Desh | Malhar, Khamaj, Desh, Marva, Kedar |
| LAVANI_029 | Ugvali Shukrachi Chandni | Aarti Anklikar | Lavani | Ahir Bhairav | 0.9206 | Desh | 0.9204 | 0.0003 | Ahir Bhairav, Desh, Puriya | Ahir Bhairav, Desh, Puriya, Darbari, Barwa |
| LAVANI_030 | Ghotala Ghotala Zala | Vaishali Samant | Lavani | Bilaskhani Todi | 0.9362 | Marva | 0.9360 | 0.0001 | Bilaskhani Todi, Marva, Desh | Bilaskhani Todi, Marva, Desh, Bageshree, Puriya |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Urmila Dhangkar | Lavani | Adana | 0.9340 | Bhoopali | 0.9327 | 0.0014 | Adana, Bhoopali, Yaman or Kalyan | Adana, Bhoopali, Yaman or Kalyan |
| LAVANI_032 | Aase Wajwa Ki | Bela Shende and chorus | Lavani | Megh | 0.9426 | Miyan Ki Todi orTodi or Darbari Todi | 0.9421 | 0.0005 | Megh, Miyan Ki Todi orTodi or Darbari Todi, Nat Bhairav | Megh, Miyan Ki Todi orTodi or Darbari Todi, Nat Bhairav, Puriya, Desh |
| LAVANI_033 | Dolyavar Gogal Lava | Various Artists | Lavani | Darbari | 0.9348 | Miyan Ki Todi orTodi or Darbari Todi | 0.9345 | 0.0002 | Darbari, Miyan Ki Todi orTodi or Darbari Todi, Purba | Darbari, Miyan Ki Todi orTodi or Darbari Todi, Purba, sarang, Kedar |
| LAVANI_034 | Reshmachya Reghani | Asha Bhosle | Lavani | Khamaj | 0.9433 | Gauri | 0.9429 | 0.0004 | Khamaj, Gauri, Basant | Khamaj, Gauri, Basant, Bahar, Malkauns |
| LAVANI_035 | Chala Jejurila Jau | Uttara Kelkar | Lavani | Yaman Kalyan | 0.9391 | Desh | 0.9385 | 0.0006 | Yaman Kalyan, Desh, Alhaiya Bilaval | Yaman Kalyan, Desh, Alhaiya Bilaval, Darbari, Bhairavi |
| MAAND_001 | Kesariya Balam Aavo Ni | Allah Jilai Bai | Maand | Darbari | 0.9456 | Bahar | 0.9423 | 0.0034 | Darbari, Bahar, Purba | Darbari, Bahar, Purba, Multani, Jaunpuri |
| MAAND_002 | Moray Aaj Aaye Saiyan | Allah Jilai Bai | Maand | Hemant | 0.9179 | Bihag | 0.9174 | 0.0005 | Hemant, Bihag, Bibhas | Hemant, Bihag, Bibhas, Lalit, Desh |
| MAAND_003 | Moomal | Allah Jilai Bai | Maand | Madhuvanti | 0.9465 | Jaunpuri | 0.9453 | 0.0012 | Madhuvanti, Jaunpuri, Multani | Madhuvanti, Jaunpuri, Multani, Desh, Purba |
| MAAND_004 | Ghoomar | Allah Jilai Bai | Maand | Purba | 0.9283 | Bilaskhani Todi | 0.9236 | 0.0047 | Purba, Bilaskhani Todi, nan | Purba, Bilaskhani Todi |
| MAAND_005 | Bai Sa Ra Beera | Allah Jilai Bai | Maand | Megh | 0.9415 | Bahar | 0.9408 | 0.0008 | Megh, Bahar, Miyan Ki Todi orTodi or Darbari Todi | Megh, Bahar, Miyan Ki Todi orTodi or Darbari Todi, Purba, Bhairav Bahar |
| MAAND_006 | Jawai Balha Lagoji | Allah Jilai Bai | Maand | Bahar | 0.9220 | Bageshree | 0.9212 | 0.0007 | Bahar, Bageshree, Asavari | Bahar, Bageshree, Asavari, Gauri, Yaman or Kalyan |
| MAAND_007 | Kesariya Balam Aavo Ni | Gavri Devi | Maand | Darbari | 0.9233 | Madhuvanti | 0.9231 | 0.0002 | Darbari, Madhuvanti, Bahar | Darbari, Madhuvanti, Bahar, Bilaskhani Todi, Sorath |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Seema Mishra | Maand | Bhatiyar | 0.9289 | Miyan Ki Todi orTodi or Darbari Todi | 0.9287 | 0.0002 | Bhatiyar, Miyan Ki Todi orTodi or Darbari Todi, Madhuvanti | Bhatiyar, Miyan Ki Todi orTodi or Darbari Todi, Madhuvanti, Poorvi, Hemant |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Seema Mishra | Maand | Khamaj | 0.9444 | Ahir Bhairav | 0.9431 | 0.0013 | Khamaj, Ahir Bhairav, Bhoopali | Khamaj, Ahir Bhairav, Bhoopali, Asavari, Sohni |
| MAAND_010 | Badila Yahi Rahiyo Saa | Banarsi Babu | Maand | Gauri | 0.9537 | Bageshree | 0.9457 | 0.0080 | Gauri, Bageshree, Adana | Gauri, Bageshree, Adana, Desh, Khamaj |
| MAAND_011 | Taniya Resham Ki | Banarsi Babu | Maand | Adana | 0.9454 | Khamaj | 0.9430 | 0.0024 | Adana, Khamaj, Alhaiya Bilaval | Adana, Khamaj, Alhaiya Bilaval |
| MAAND_012 | Kad Janulee Sasre | Rehana Mirza | Maand | Alhaiya Bilaval | 0.9421 | Bhairavi | 0.9388 | 0.0034 | Alhaiya Bilaval, Bhairavi, Darbari | Alhaiya Bilaval, Bhairavi, Darbari, Bhairav, Bahar |
| MAAND_013 | Le Dekh Nangi Melo | Rehana Mirza | Maand | Khamaj | 0.9477 | Malhar | 0.9466 | 0.0011 | Khamaj, Malhar, Megh Malhar | Khamaj, Malhar, Megh Malhar, Purba, Yaman Kalyan |
| MAAND_014 | The To Piyo Sadarudi | Rehana Mirza | Maand | Rageshree | 0.9392 | Abhogi | 0.9385 | 0.0007 | Rageshree, Abhogi, Bilaskhani Todi | Rageshree, Abhogi, Bilaskhani Todi, Marva, Durga |
| MAAND_015 | Kesariya Balam (Desert Slide) | Anwar Khan | Maand | Miyan Ki Todi orTodi or Darbari Todi | 0.9267 | Malkauns | 0.9239 | 0.0029 | Miyan Ki Todi orTodi or Darbari Todi, Malkauns, Darbari | Miyan Ki Todi orTodi or Darbari Todi, Malkauns, Darbari, Sohni, Madhuvanti |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Anwar Khan | Maand | Bhairavi | 0.9186 | Madhuvanti | 0.9147 | 0.0039 | Bhairavi, Madhuvanti, Pahadi | Bhairavi, Madhuvanti, Pahadi, Purba, Hemant |
| MAAND_017 | Apne Desh Ri Aajadi | Anwar Khan | Maand | Sorath | 0.9351 | Basant | 0.9312 | 0.0038 | Sorath, Basant, Adana | Sorath, Basant, Adana |
| MAAND_018 | Kesariya Balam Aao Ni | Anwar Khan | Maand | Ahir Bhairav | 0.9354 | Gauri | 0.9319 | 0.0035 | Ahir Bhairav, Gauri, Sohni | Ahir Bhairav, Gauri, Sohni, Khamaj, Basant |
| MAAND_019 | Raag Maand Bandish | Mame Khan | Maand | Bilaval | 0.9388 | Purba | 0.9388 | 0.0000 | Bilaval, Purba, Sorath | Bilaval, Purba, Sorath, Khamaj, Adana |
| MAAND_020 | Rajasthani Folk Raag Maand | Mame Khan | Maand | Bilaval | 0.9409 | Purba | 0.9397 | 0.0012 | Bilaval, Purba, Sorath | Bilaval, Purba, Sorath, Khamaj, suddha kalyan |
| MAAND_021 | Chaudhary | Mame Khan | Maand | Ahir Bhairav | 0.9324 | Desh | 0.9307 | 0.0017 | Ahir Bhairav, Desh, Nat Bhairav | Ahir Bhairav, Desh, Nat Bhairav, Bhairavi, suddha kalyan |
| MAAND_022 | Chhap Tilak Sab Cheeni | Kirti Sagathia | Maand | Adana | 0.9323 | Bahar | 0.9323 | 0.0001 | Adana, Bahar, Darbari | Adana, Bahar, Darbari, Nat Bhairav, suddha kalyan |
| MAAND_023 | Balam | Swaroop Khan | Maand | Darbari | 0.9495 | Asavari | 0.9451 | 0.0044 | Darbari, Asavari, nan | Darbari, Asavari |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Swaroop Khan Sura | Maand | Marva | 0.9147 | Ahir Bhairav | 0.9136 | 0.0011 | Marva, Ahir Bhairav, Barwa | Marva, Ahir Bhairav, Barwa, Bhairavi, Bibhas |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Sikander Khan Baru | Maand | Khamaj | 0.9313 | Adana | 0.9296 | 0.0017 | Khamaj, Adana, Bahar | Khamaj, Adana, Bahar, Pahadi, Miyan Ki Todi orTodi or Darbari Todi |
| MAAND_026 | Darude Ro Rog Lagiyo | Sikander Khan Baru | Maand | Ahir Bhairav | 0.9131 | Puriya | 0.9098 | 0.0033 | Ahir Bhairav, Puriya, Darbari | Ahir Bhairav, Puriya, Darbari, Asavari |
| MAAND_027 | Kesariya Balam Padharo | Bhungar Khan | Maand | Komal Bhairav | 0.9439 | Bhoopali | 0.9401 | 0.0038 | Komal Bhairav, Bhoopali, Bhimpalasi | Komal Bhairav, Bhoopali, Bhimpalasi, Khamaj, Purba |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Kheta Khan, Fakira Khan Bhadresh | Maand | Kafi | 0.9416 | Pahadi | 0.9397 | 0.0019 | Kafi, Pahadi, Desh | Kafi, Pahadi, Desh, Bilaskhani Todi, Malhar |
| MAAND_029 | Kalali | Latif Khan | Maand | Malkauns | 0.9186 | Bhairavi | 0.9169 | 0.0017 | Malkauns, Bhairavi, Megh | Malkauns, Bhairavi, Megh, Desh, Ahir Bhairav |
| MAAND_030 | Kesariya Balam | Raitila (Group) | Maand | Yaman Kalyan | 0.9278 | Sorath | 0.9263 | 0.0014 | Yaman Kalyan, Sorath, Purba | Yaman Kalyan, Sorath, Purba, Malkauns, Puriya |
| MAAND_031 | Joori Thaane Na Maare Raad | Jameela Bai & Kulsum Bai | Maand | Yaman Kalyan | 0.9218 | Bhimpalasi | 0.9212 | 0.0007 | Yaman Kalyan, Bhimpalasi, Bilaskhani Todi | Yaman Kalyan, Bhimpalasi, Bilaskhani Todi, Lalit, Basant |
| MAAND_032 | Alija Dhola Ekladi | Jameela Bai & Kulsum Bai | Maand | Desh | 0.9228 | Bhimpalasi | 0.9190 | 0.0037 | Desh, Bhimpalasi, Yaman Kalyan | Desh, Bhimpalasi, Yaman Kalyan, Bhairavi |
| MAAND_033 | Mumal | Jameela Bai & Kulsum Bai | Maand | Purba | 0.9342 | Yaman Kalyan | 0.9331 | 0.0012 | Purba, Yaman Kalyan, Bhimpalasi | Purba, Yaman Kalyan, Bhimpalasi, Bhatiyar, Bilaskhani Todi |
| MAAND_034 | Mhari Sajni Ae | Nand Lal Bhat | Maand | Sorath | 0.9439 | Bhairavi | 0.9420 | 0.0019 | Sorath, Bhairavi, Yaman Kalyan | Sorath, Bhairavi, Yaman Kalyan, Puriya, Darbari |
| MAAND_035 | Rajwadi Mehfil | Nand Lal Bhat | Maand | Alhaiya Bilaval | 0.9365 | Marva | 0.9363 | 0.0002 | Alhaiya Bilaval, Marva, Megh | Alhaiya Bilaval, Marva, Megh, Bhairavi, Basant |
| MAAND_036 | Thane Samjhaye Sakha Sa | Munshi Khan | Maand | Darbari | 0.9420 | Puriya | 0.9395 | 0.0025 | Darbari, Puriya, Bibhas | Darbari, Puriya, Bibhas, Adana, Basant |
| MAAND_037 | Padharo Mhare Mehal | Munshi Khan | Maand | Bhupal Todi | 0.9445 | Ramkali | 0.9417 | 0.0027 | Bhupal Todi, Ramkali, Komal Bhairav | Bhupal Todi, Ramkali, Komal Bhairav, Bibhas, Adana |
| MAAND_038 | Daru Dakha Ro | Munshi Khan | Maand | Gauri | 0.9424 | Alhaiya Bilaval | 0.9399 | 0.0025 | Gauri, Alhaiya Bilaval, Malhar | Gauri, Alhaiya Bilaval, Malhar, Khamaj, Kafi |
| MAAND_039 | Mahara Sajan Base | Munshi Khan | Maand | Purba | 0.9428 | Desh | 0.9375 | 0.0053 | Purba, Desh, Yaman Kalyan | Purba, Desh, Yaman Kalyan, Adana, Bhatiyar |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Human Sagar | Sambalpuri | Ahir Bhairav | 0.9407 | Malkauns | 0.9363 | 0.0044 | Ahir Bhairav, Malkauns, Pahadi | Ahir Bhairav, Malkauns, Pahadi, Bhimpalasi, Desh |
| SAMBALPURI_002 | Tor Lagi Marijibi | Human Sagar | Sambalpuri | Bilaval | 0.9375 | Ahir Bhairav | 0.9366 | 0.0008 | Bilaval, Ahir Bhairav, Sorath | Bilaval, Ahir Bhairav, Sorath, Darbari, Desh |
| SAMBALPURI_003 | Sacha Lover | Human Sagar | Sambalpuri | Basant | 0.9391 | Darbari | 0.9370 | 0.0021 | Basant, Darbari, Asavari | Basant, Darbari, Asavari, Sorath, Marva |
| SAMBALPURI_004 | Pardesia Raja | Human Sagar | Sambalpuri | Sorath | 0.9508 | Darbari | 0.9491 | 0.0017 | Sorath, Darbari, Bageshree | Sorath, Darbari, Bageshree, Desh, Miyan Ki Todi orTodi or Darbari Todi |
| SAMBALPURI_005 | Lal Sadhi | Human Sagar | Sambalpuri | Bilaval | 0.9279 | Desh | 0.9254 | 0.0025 | Bilaval, Desh, Bageshree | Bilaval, Desh, Bageshree, Bhairavi, Kedar |
| SAMBALPURI_006 | Rasia | Mantu Chhuria | Sambalpuri | Bilaval | 0.9351 | Bageshree | 0.9276 | 0.0074 | Bilaval, Bageshree, Bhimpalasi | Bilaval, Bageshree, Bhimpalasi, Asavari, Desh |
| SAMBALPURI_007 | A Mo Dil | Mantu Chhuria | Sambalpuri | Darbari | 0.9364 | Bageshree | 0.9330 | 0.0034 | Darbari, Bageshree, Adana | Darbari, Bageshree, Adana, Asavari, Malkauns |
| SAMBALPURI_008 | Hae Go | Mantu Chhuria | Sambalpuri | Darbari | 0.9347 | Megh | 0.9345 | 0.0002 | Darbari, Megh, Sorath | Darbari, Megh, Sorath, Bilaval, Ahir Bhairav |
| SAMBALPURI_009 | Aukaat | Mantu Chhuria | Sambalpuri | Megh | 0.9504 | Bhatiyar | 0.9495 | 0.0008 | Megh, Bhatiyar, Sohni | Megh, Bhatiyar, Sohni, Bhimpalasi, Asavari |
| SAMBALPURI_010 | Rani Guri | Mantu Chhuria | Sambalpuri | Bageshree | 0.9397 | Puriya | 0.9367 | 0.0030 | Bageshree, Puriya, Alhaiya Bilaval | Bageshree, Puriya, Alhaiya Bilaval, Adana, Bhoopali |
| SAMBALPURI_011 | Gidja Gija Re | Asima Panda | Sambalpuri | Khamaj | 0.9379 | Bageshree | 0.9376 | 0.0003 | Khamaj, Bageshree, Bilaval | Khamaj, Bageshree, Bilaval, Adana, Abhogi |
| SAMBALPURI_012 | Tui Bewafa | Asima Panda | Sambalpuri | Desh | 0.9439 | Adana | 0.9402 | 0.0037 | Desh, Adana, Miyan Ki Todi orTodi or Darbari Todi | Desh, Adana, Miyan Ki Todi orTodi or Darbari Todi, Sorath, Ahir Bhairav |
| SAMBALPURI_013 | Ramagaru | Asima Panda | Sambalpuri | Sorath | 0.9276 | Desh | 0.9275 | 0.0001 | Sorath, Desh, Alhaiya Bilaval | Sorath, Desh, Alhaiya Bilaval |
| SAMBALPURI_014 | Deewana Banalu Rani | Asima Panda | Sambalpuri | Bageshree | 0.9338 | Megh | 0.9335 | 0.0002 | Bageshree, Megh, Rageshree | Bageshree, Megh, Rageshree, Ahir Bhairav, Adana |
| SAMBALPURI_015 | SambalPuri Saiyaan | Asima Panda | Sambalpuri | Sorath | 0.9390 | Desh | 0.9374 | 0.0016 | Sorath, Desh, Adana | Sorath, Desh, Adana, Miyan Ki Todi orTodi or Darbari Todi, Ahir Bhairav |
| SAMBALPURI_016 | Anta Halei Halei | Ruku Suna | Sambalpuri | Alhaiya Bilaval | 0.9476 | Bhairavi | 0.9467 | 0.0009 | Alhaiya Bilaval, Bhairavi, Pahadi | Alhaiya Bilaval, Bhairavi, Pahadi, Darbari, Ahir Bhairav |
| SAMBALPURI_017 | Chandir Paenjhal | Ruku Suna | Sambalpuri | Adana | 0.9446 | Asavari | 0.9443 | 0.0003 | Adana, Asavari, Desh | Adana, Asavari, Desh, Alhaiya Bilaval, Darbari |
| SAMBALPURI_018 | Prem Paga | Ruku Suna | Sambalpuri | Darbari | 0.9492 | Asavari | 0.9469 | 0.0023 | Darbari, Asavari, Adana | Darbari, Asavari, Adana, Ahir Bhairav, Pahadi |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Ruku Suna | Sambalpuri | Megh | 0.9427 | Hindol | 0.9419 | 0.0007 | Megh, Hindol, Alhaiya Bilaval | Megh, Hindol, Alhaiya Bilaval, Adana, Nat Bhairav |
| SAMBALPURI_020 | A Labanga Lata | Ruku Suna | Sambalpuri | Bhairavi | 0.9297 | Bhoopali | 0.9291 | 0.0006 | Bhairavi, Bhoopali, Asavari | Bhairavi, Bhoopali, Asavari, Abhogi, Pahadi |
| SAMBALPURI_021 | Badnam | Prakash Jal | Sambalpuri | Rageshree | 0.9225 | Kedar | 0.9185 | 0.0040 | Rageshree, Kedar, Pahadi | Rageshree, Kedar, Pahadi, Ahir Bhairav, Yaman or Kalyan |
| SAMBALPURI_022 | Dil Bhitre Lekhideli | Prakash Jal | Sambalpuri | Pahadi | 0.9342 | Ahir Bhairav | 0.9319 | 0.0023 | Pahadi, Ahir Bhairav, nan | Pahadi, Ahir Bhairav |
| SAMBALPURI_023 | Bewafa Mita | Prakash Jal | Sambalpuri | Adana | 0.9353 | Durga | 0.9340 | 0.0013 | Adana, Durga, Bhoopali | Adana, Durga, Bhoopali, Pahadi, Marva |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Prakash Jal | Sambalpuri | Pahadi | 0.9374 | Nat Bhairav | 0.9368 | 0.0006 | Pahadi, Nat Bhairav, Miyan Ki Todi orTodi or Darbari Todi | Pahadi, Nat Bhairav, Miyan Ki Todi orTodi or Darbari Todi, Bihag, Ahir Bhairav |
| SAMBALPURI_025 | Le Le Baula | Antara Chakrabarty | Sambalpuri | Asavari | 0.9484 | Adana | 0.9467 | 0.0017 | Asavari, Adana, Alhaiya Bilaval | Asavari, Adana, Alhaiya Bilaval, Puriya, Ahir Bhairav |
| SAMBALPURI_027 | Feel My Love | Antara Chakrabarty | Sambalpuri | Marva | 0.9430 | Madhuvanti | 0.9425 | 0.0005 | Marva, Madhuvanti, Bhimpalasi | Marva, Madhuvanti, Bhimpalasi, Desh, Darbari |
| SAMBALPURI_028 | Kulfirani Chocobar | Ira Mohanty | Sambalpuri | Marva | 0.9496 | Megh | 0.9429 | 0.0067 | Marva, Megh, Asavari | Marva, Megh, Asavari, Alhaiya Bilaval |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Ira Mohanty | Sambalpuri | Khamaj | 0.9419 | Bageshree | 0.9366 | 0.0053 | Khamaj, Bageshree, Asavari | Khamaj, Bageshree, Asavari, Adana, Darbari |
| SAMBALPURI_030 | Sundri Janha | Ira Mohanty | Sambalpuri | Bageshree | 0.9371 | Asavari | 0.9356 | 0.0015 | Bageshree, Asavari, Ahir Bhairav | Bageshree, Asavari, Ahir Bhairav, Kedar, Darbari |
| SAMBALPURI_031 | Rangabati | Jitendra Haripal | Sambalpuri | Bahar | 0.9151 | Kedar | 0.9113 | 0.0038 | Bahar, Kedar, Adana | Bahar, Kedar, Adana, Rageshree, Bhairav |
| SAMBALPURI_032 | Aha Mor Galar Malire | Jitendra Haripal | Sambalpuri | Basant | 0.9320 | Marva | 0.9267 | 0.0052 | Basant, Marva, Ahir Bhairav | Basant, Marva, Ahir Bhairav, Sorath, Malkauns |
| SAMBALPURI_033 | Din Gani Gani | Jitendra Haripal | Sambalpuri | Shree | 0.9340 | Alhaiya Bilaval | 0.9339 | 0.0001 | Shree, Alhaiya Bilaval, Khamaj | Shree, Alhaiya Bilaval, Khamaj, Desi, Asavari |
| SAMBALPURI_034 | Namo Namo Ago | Jitendra Haripal | Sambalpuri | Bhairav | 0.9314 | Kedar | 0.9306 | 0.0009 | Bhairav, Kedar, Rageshree | Bhairav, Kedar, Rageshree, Bhoopali, Puriya |