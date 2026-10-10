# Complete Folk Music Raag Prediction Report

## Method
Folk audio → CultureMERT-95M → 5-second 768-D embeddings → 60-second temporal grouping → Mean pooling → L2 normalization → Cosine similarity against TRF reference embeddings → Recording-level aggregation → Candidate Raag prediction.

These are **retrieved candidate Raag associations** and NOT proven ground-truth Raag labels.

## PART 1 — COMPLETE OVERVIEW TABLE

| No. | Original Song ID | Song Name | Artist | Genre | Status | Predicted Raag | Top-1 Similarity | Top-2 Raag | Top-2 Similarity | Margin | Top-3 | Top-4 | Top-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BAUL_001 | Agun Pani | Purna Das Baul | Baul | Successfully classified | Madhuvanti | 0.9390 | Darbari | 0.9383 | 0.0008 | Bhairavi | Purba | Bahar |
| 2 | BAUL_002 | Ja Lolite Anga Kushum Tule | Purna Das Baul | Baul | Successfully classified | Abhogi | 0.9225 | Malkauns | 0.9201 | 0.0025 | Darbari | Yaman Kalyan | Alhaiya Bilaval |
| 3 | BAUL_003 | Kalkatou Mon Heshekeley | Purna Das Baul | Baul | Successfully classified | Yaman Kalyan | 0.9422 | Malkauns | 0.9393 | 0.0029 | Abhogi | Madhuvanti | Darbari |
| 4 | BAUL_004 | Ananda Lahari | Purna Das Baul | Baul | Successfully classified | Marva | 0.9303 | Bhairavi | 0.9264 | 0.0039 | Khamaj | Malhar | Darbari |
| 5 | BAUL_005 | Amar Aei Rudir Dhara | Purna Das Baul | Baul | Successfully classified | Bhairavi | 0.9324 | Alhaiya Bilaval | 0.9293 | 0.0031 | Malkauns | Purba | sarang |
| 6 | BAUL_006 | Hari Balo Re | Purna Das Baul | Baul | Successfully classified | Yaman Kalyan | 0.9325 | Malkauns | 0.9291 | 0.0033 | Puriya | Darbari | Sorath |
| 7 | BAUL_007 | Gour Lila Bazare | Purna Das Baul | Baul | Successfully classified | Yaman Kalyan | 0.9373 | Megh | 0.9314 | 0.0059 | Darbari | Yaman or Kalyan | Kafi |
| 8 | BAUL_008 | Tripti | Purna Das Baul | Baul | Successfully classified | Malkauns | 0.8812 | Sorath | 0.8810 | 0.0002 | Basant | Bhairavi | Malhar |
| 9 | BAUL_009 | Kon Ba Zharir | Purna Das Baul | Baul | Successfully classified | Barwa | 0.9362 | Lalit | 0.9350 | 0.0012 | Pahadi | Desh | Puriya |
| 10 | BAUL_010 | Sonogo Aian Dada | Purna Das Baul | Baul | Successfully classified | Yaman Kalyan | 0.9457 | Bahar | 0.9417 | 0.0040 | Malkauns | Bhimpalasi | Asavari |
| 11 | BAUL_011 | Chor Poreche Babur Bagane | Purna Das Baul | Baul | Successfully classified | Bhairavi | 0.9281 | Megh | 0.9205 | 0.0076 | Gauri | Alhaiya Bilaval | Abhogi |
| 12 | BAUL_012 | Golemale Gole Male | Purna Das Baul | Baul | Successfully classified | Khamaj | 0.9357 | Jhinjhoti | 0.9341 | 0.0016 | Abhogi | Bilaval | Purba |
| 13 | BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Purna Das Baul | Baul | Successfully classified | Puriya | 0.9328 | Jaunpuri | 0.9276 | 0.0052 | Pahadi | Sohni | N/A |
| 14 | BAUL_014 | Dekhechhi Rup-Sagare | Purna Das Baul | Baul | Successfully classified | Bhairavi | 0.9367 | Sorath | 0.9317 | 0.0049 | N/A | N/A | N/A |
| 15 | BAUL_015 | Aami Kothay Pabo Tare | Purna Das Baul | Baul | Successfully classified | Abhogi | 0.9181 | Bhairavi | 0.9172 | 0.0009 | Malkauns | Asavari | Kedar |
| 16 | BAUL_016 | Shesher Deene Sejon Bine | Purna Das Baul | Baul | Successfully classified | Sorath | 0.9355 | Madhuvanti | 0.9304 | 0.0051 | Darbari | Malkauns | Purba |
| 17 | BAUL_017 | Milon Hobe Koto Dine | Farida Parveen | Baul | Successfully classified | Bhimpalasi | 0.9335 | Bhoopali | 0.9318 | 0.0017 | Barwa | Desh | Pahadi |
| 18 | BAUL_018 | Khanchar Bhitor Achin Pakhi | Farida Parveen | Baul | Successfully classified | Abhogi | 0.9294 | Yaman or Kalyan | 0.9289 | 0.0005 | Kafi | Jhinjhoti | Adana |
| 19 | BAUL_019 | Aami Ek Pagal Baul | Purna Das Baul | Baul | Successfully classified | Bhairavi | 0.9208 | Ahir Bhairav | 0.9207 | 0.0001 | Sorath | Nat Bhairav | Darbari |
| 20 | BAUL_020 | Jemon Beni Temni Rabey | Purna Das Baul | Baul | Successfully classified | Bhairav | 0.9400 | Ramkali | 0.9357 | 0.0043 | Khamaj | Desh | Megh Malhar |
| 21 | BAUL_021 | Prem Koraa Ki Jwalaa | Purna Das Baul | Baul | Successfully classified | Purba | 0.9417 | Gauri | 0.9334 | 0.0084 | Bageshree | N/A | N/A |
| 22 | BAUL_022 | Emon Mon Byabsaa | Purna Das Baul | Baul | Successfully classified | Durga | 0.9488 | Gauri | 0.9449 | 0.0040 | Rageshree | Asavari | Bihag |
| 23 | BAUL_023 | Manush Bhajo Manusheyi Pujo | Purna Das Baul | Baul | Successfully classified | Purba | 0.9455 | N/A | 0.0000 | 0.9455 | N/A | N/A | N/A |
| 24 | BAUL_024 | Dil Dariyar Majhete | Purna Das Baul | Baul | Successfully classified | Purba | 0.9334 | Bageshree | 0.9246 | 0.0088 | Desh | N/A | N/A |
| 25 | BAUL_025 | Bahu Punya Phale Krishna Mele | Purna Das Baul | Baul | Successfully classified | Purba | 0.9353 | Darbari | 0.9294 | 0.0059 | Rageshree | Kedar | N/A |
| 26 | BAUL_026 | Sokhi Jamuna Jol | Purna Das Baul | Baul | Successfully classified | Malkauns | 0.9320 | Megh | 0.9315 | 0.0005 | Yaman or Kalyan | Yaman Kalyan | Desh |
| 27 | BAUL_027 | Pre Loye Jao Amay | Farida Parveen | Baul | Successfully classified | Malkauns | 0.9434 | Bhairav | 0.9402 | 0.0032 | Poorvi | Abhogi | Bhoopali |
| 28 | BAUL_028 | Somoy Gele Sadhon Hobena | Farida Parveen | Baul | Successfully classified | Nat Bhairav | 0.9180 | Megh | 0.9171 | 0.0009 | Bageshree | Khamaj | N/A |
| 29 | BAUL_029 | Aar Amare Marishne Maa | Farida Parveen | Baul | Successfully classified | Miyan Ki Todi orTodi or Darbari Todi | 0.9414 | Bahar | 0.9394 | 0.0020 | Desh | Bilaskhani Todi | Pahadi |
| 30 | BAUL_030 | Dhonno Dhonno Boli Tare | Farida Parveen | Baul | Successfully classified | Bahar | 0.9370 | Miyan Ki Todi orTodi or Darbari Todi | 0.9339 | 0.0031 | Bihag | Bageshree | Yaman Kalyan |
| 31 | BHANGRA_001 | Tera Yaar Bolda | Surjit Bindrakhia | Bhangra | Successfully classified | Bahar | 0.9212 | Darbari | 0.9210 | 0.0001 | Alhaiya Bilaval | Bibhas | Bhoopali |
| 32 | BHANGRA_002 | Jatt Di Pasand | Surjit Bindrakhia | Bhangra | Successfully classified | Bhoopali | 0.9307 | Bahar | 0.9272 | 0.0035 | Asavari | Darbari | Yaman or Kalyan |
| 33 | BHANGRA_003 | Dupatta Tera Satrang Da | Surjit Bindrakhia | Bhangra | Successfully classified | Darbari | 0.9340 | Alhaiya Bilaval | 0.9340 | 0.0000 | Bibhas | Bahar | N/A |
| 34 | BHANGRA_004 | Wang Wargi Kudi | Surjit Bindrakhia | Bhangra | Successfully classified | Darbari | 0.9286 | Abhogi | 0.9243 | 0.0043 | Ahir Bhairav | N/A | N/A |
| 35 | BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Surjit Bindrakhia | Bhangra | Successfully classified | Yaman or Kalyan | 0.9279 | Megh | 0.9239 | 0.0041 | Bahar | Khamaj | Marva |
| 36 | BHANGRA_006 | Munda Ki Mangda | Surjit Bindrakhia | Bhangra | Successfully classified | Adana | 0.9311 | Alhaiya Bilaval | 0.9282 | 0.0029 | Bhairav Bahar | Jaunpuri | N/A |
| 37 | BHANGRA_007 | Tunak Tunak Tun | Daler Mehndi | Bhangra | Successfully classified | Megh | 0.9487 | Deepak | 0.9359 | 0.0127 | Bhatiyar | Bilaskhani Todi | sarang |
| 38 | BHANGRA_008 | Bolo Ta Ra Ra | Daler Mehndi | Bhangra | Successfully classified | Desh | 0.9417 | Pahadi | 0.9385 | 0.0032 | Asavari | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 39 | BHANGRA_009 | Ho Jayegi Balle Balle | Daler Mehndi | Bhangra | Successfully classified | Darbari | 0.9468 | Basant | 0.9397 | 0.0071 | Asavari | Kedar | N/A |
| 40 | BHANGRA_010 | Dardi Rab Rab Kardi | Daler Mehndi | Bhangra | Successfully classified | Darbari | 0.9320 | Asavari | 0.9275 | 0.0045 | Alhaiya Bilaval | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 41 | BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Daler Mehndi | Bhangra | Successfully classified | Adana | 0.9344 | Ahir Bhairav | 0.9298 | 0.0046 | Barwa | Deepak | Bibhas |
| 42 | BHANGRA_012 | Na Na Na Na Na Re | Daler Mehndi | Bhangra | Successfully classified | Adana | 0.9397 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | 0.0015 | Madhuvanti | Rageshree | Alhaiya Bilaval |
| 43 | BHANGRA_013 | Pao Bhangra | Balwinder Safri | Bhangra | Successfully classified | Bhairavi | 0.9151 | Bahar | 0.9140 | 0.0011 | Malhar | Megh Malhar | Asavari |
| 44 | BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Balwinder Safri | Bhangra | Successfully classified | Gujari Todi | 0.9384 | Megh | 0.9304 | 0.0081 | Puriya | Bilaskhani Todi | Miyan Ki Todi orTodi or Darbari Todi |
| 45 | BHANGRA_015 | Sariyan Toon Sohni | Balwinder Safri | Bhangra | Successfully classified | Alhaiya Bilaval | 0.9291 | Darbari | 0.9286 | 0.0005 | Asavari | Puriya | Ahir Bhairav |
| 46 | BHANGRA_016 | Mitran Ne Mundri Te | Balwinder Safri | Bhangra | Successfully classified | Darbari | 0.9400 | Puriya | 0.9394 | 0.0006 | Alhaiya Bilaval | Asavari | N/A |
| 47 | BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Balwinder Safri | Bhangra | Successfully classified | Gujari Todi | 0.9387 | Bilaval | 0.9333 | 0.0054 | Puriya | Megh | Alhaiya Bilaval |
| 48 | BHANGRA_018 | Tutak Tutak Tutiya | Malkit Singh (Golden Star) | Bhangra | Successfully classified | Darbari | 0.9297 | Alhaiya Bilaval | 0.9295 | 0.0002 | Desh | Asavari | Gujari Todi |
| 49 | BHANGRA_019 | Kurri Garam Jayee | Malkit Singh (Golden Star) | Bhangra | Successfully classified | Rageshree | 0.9322 | Puriya | 0.9319 | 0.0003 | Gauri | Kedar | N/A |
| 50 | BHANGRA_020 | Chal Hun | Malkit Singh (Golden Star) | Bhangra | Successfully classified | Shree | 0.9341 | Gujari Todi | 0.9327 | 0.0014 | Bhimpalasi | Darbari | Madhuvanti |
| 51 | BHANGRA_021 | Kinni Sohni | Malkit Singh (Golden Star) | Bhangra | Successfully classified | Darbari | 0.9421 | Alhaiya Bilaval | 0.9358 | 0.0063 | Rageshree | Adana | Desi |
| 52 | BHANGRA_022 | Mera Yaar Vajaye Dhol | Apna Sangeet | Bhangra | Successfully classified | Bilaval | 0.9433 | Malkauns | 0.9424 | 0.0009 | Yaman Kalyan | Sorath | Purba |
| 53 | BHANGRA_023 | Bhaina Nachiyaan | Apna Sangeet | Bhangra | Successfully classified | Alhaiya Bilaval | 0.9301 | Puriya | 0.9252 | 0.0049 | Asavari | Barwa | N/A |
| 54 | BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Apna Sangeet | Bhangra | Successfully classified | Bilaval | 0.9226 | Desh | 0.9145 | 0.0081 | Ahir Bhairav | Bageshree | Darbari |
| 55 | BHANGRA_025 | Soho Road Utey | Apna Sangeet | Bhangra | Successfully classified | Malkauns | 0.9347 | Bilaval | 0.9321 | 0.0026 | Desh | Darbari | Khamaj |
| 56 | BHANGRA_026 | Kuriyan Ne Paya Bhangra | Kuldeep Manak | Bhangra | Successfully classified | Darbari | 0.9309 | Asavari | 0.9244 | 0.0065 | Alhaiya Bilaval | Madhuvanti | N/A |
| 57 | BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Kuldeep Manak | Bhangra | Successfully classified | Darbari | 0.9342 | Asavari | 0.9312 | 0.0030 | N/A | N/A | N/A |
| 58 | BHANGRA_028 | Jugni | Kuldeep Manak | Bhangra | Successfully classified | Yaman Kalyan | 0.9351 | Sorath | 0.9308 | 0.0043 | Adana | Bhimpalasi | Desh |
| 59 | BHANGRA_029 | Putt Jattan De | Surinder Shinda | Bhangra | Successfully classified | Abhogi | 0.9170 | Darbari | 0.9168 | 0.0002 | Pahadi | Bhairavi | Desi |
| 60 | BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Surinder Shinda | Bhangra | Successfully classified | suddha kalyan | 0.9264 | Purba | 0.9229 | 0.0035 | Adana | Bilaskhani Todi | N/A |
| 61 | BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Surinder Shinda | Bhangra | Successfully classified | Rageshree | 0.9379 | Khamaj | 0.9289 | 0.0090 | Bilaval | Sorath | Bhairavi |
| 62 | BHANGRA_032 | Jeona Morh | Surinder Shinda | Bhangra | Successfully classified | Basant Mukhari | 0.9448 | Bhairav | 0.9441 | 0.0007 | Basant | Megh Malhar | Malhar |
| 63 | BIHU_001 | Morom Logai Logai | Zubeen Garg | Bihu | Successfully classified | Sohni | 0.9355 | Bahar | 0.9334 | 0.0021 | Megh Malhar | Bhimpalasi | suddha kalyan |
| 64 | BIHU_002 | Toi Nagini Ne | Zubeen Garg | Bihu | Successfully classified | Asavari | 0.9280 | Puriya | 0.9271 | 0.0009 | Sohni | Bahar | Khamaj |
| 65 | BIHU_003 | Kokalote Tongali | Zubeen Garg | Bihu | Successfully classified | Hindol | 0.9402 | Sorath | 0.9400 | 0.0002 | Madhuvanti | suddha kalyan | Khamaj |
| 66 | BIHU_004 | Tumake Logot | Zubeen Garg | Bihu | Successfully classified | Bilaval | 0.9349 | Puriya | 0.9349 | 0.0000 | Malkauns | Darbari | Abhogi |
| 67 | BIHU_005 | Kinu Sawonire Sala | Zubeen Garg | Bihu | Successfully classified | Desi | 0.9378 | Darbari | 0.9354 | 0.0024 | Hindol | Desh | Ahir Bhairav |
| 68 | BIHU_006 | Dikhow Noi Eribo | Zubeen Garg | Bihu | Successfully classified | Desh | 0.9435 | Sorath | 0.9399 | 0.0036 | Marva | N/A | N/A |
| 69 | BIHU_007 | Phool Phulise | Zubeen Garg | Bihu | Successfully classified | Desh | 0.9279 | Pahadi | 0.9243 | 0.0037 | Malhar | Darbari | Desi |
| 70 | BIHU_008 | Porbotor Dhekiya | Zubeen Garg | Bihu | Successfully classified | Desh | 0.9392 | Desi | 0.9379 | 0.0013 | Nat Bhairav | Madhuvanti | Barwa |
| 71 | BIHU_009 | O Nahor | Zubeen Garg | Bihu | Successfully classified | Marva | 0.9373 | Bahar | 0.9318 | 0.0055 | Puriya | Bhimpalasi | Ahir Bhairav |
| 72 | BIHU_010 | Janmoni Xopunote Ahiba | Zubeen Garg | Bihu | Successfully classified | Bilaskhani Todi | 0.9465 | Purba | 0.9454 | 0.0010 | Hindol | Asavari | Desh |
| 73 | BIHU_011 | Eijon Korenu Nasoni | Bhupen Hazarika | Bihu | Successfully classified | Adana | 0.9127 | Bhimpalasi | 0.9113 | 0.0014 | Rageshree | suddha kalyan | Komal Bhairav |
| 74 | BIHU_012 | Laje Moru Moru | Bhupen Hazarika | Bihu | Successfully classified | Bhoopali | 0.9350 | Puriya | 0.9348 | 0.0002 | Darbari | Asavari | Adana |
| 75 | BIHU_013 | Ki Sai Aso | Bhupen Hazarika | Bihu | Successfully classified | Ahir Bhairav | 0.9378 | Asavari | 0.9377 | 0.0001 | Adana | Bhoopali | Puriya |
| 76 | BIHU_014 | Bihureno Birina | Bhupen Hazarika | Bihu | Successfully classified | Marva | 0.9399 | Bahar | 0.9386 | 0.0013 | Khamaj | Rageshree | Ahir Bhairav |
| 77 | BIHU_015 | Duetar Boraghor | Bhupen Hazarika | Bihu | Successfully classified | Yaman Kalyan | 0.9327 | Asavari | 0.9314 | 0.0013 | Jhinjhoti | Darbari | Bhoopali |
| 78 | BIHU_016 | Sangor Chira Pitha | Bhupen Hazarika | Bihu | Successfully classified | Adana | 0.9341 | Yaman or Kalyan | 0.9337 | 0.0004 | Bhoopali | Puriya | Asavari |
| 79 | BIHU_017 | Kopou Phul Phulile | Khagen Mahanta | Bihu | Successfully classified | Bhoopali | 0.9308 | Puriya | 0.9258 | 0.0050 | Barwa | Megh | Asavari |
| 80 | BIHU_018 | Amona Dhanoke | Khagen Mahanta | Bihu | Successfully classified | Desh | 0.9321 | Megh | 0.9310 | 0.0012 | Khamaj | Asavari | Bhupal Todi |
| 81 | BIHU_019 | O Dhei Oi | Khagen Mahanta | Bihu | Successfully classified | Megh | 0.9416 | Bhairavi | 0.9335 | 0.0081 | Desh | Bahar | Bageshree |
| 82 | BIHU_020 | Dhansiri Pare Hoi | Khagen Mahanta | Bihu | Successfully classified | Bhairavi | 0.9316 | Ahir Bhairav | 0.9304 | 0.0012 | Shree | Bhatiyar | Lalit |
| 83 | BIHU_021 | Petore Juruni Bhat | Khagen Mahanta | Bihu | Successfully classified | Megh Malhar | 0.9354 | Malhar | 0.9354 | 0.0000 | Adana | Bilaval | Bhairavi |
| 84 | BIHU_022 | Aag Fale Dekhilo | Khagen Mahanta | Bihu | Successfully classified | Bahar | 0.9400 | Deepak | 0.9375 | 0.0025 | Bhairavi | Rageshree | Bilaval |
| 85 | BIHU_023 | Aghunor Potharot | Khagen Mahanta | Bihu | Successfully classified | Bhoopali | 0.9258 | Sorath | 0.9236 | 0.0022 | sarang | Darbari | Desh |
| 86 | BIHU_024 | Ebela Hale Bau | Khagen Mahanta | Bihu | Successfully classified | Bhoopali | 0.9140 | Abhogi | 0.9121 | 0.0019 | Megh | Yaman or Kalyan | Darbari |
| 87 | BIHU_025 | Senai o Tumake Bisari | Khagen Mahanta | Bihu | Successfully classified | Ahir Bhairav | 0.9319 | Megh | 0.9283 | 0.0036 | Bahar | Bilaval | Rageshree |
| 88 | BIHU_026 | Bohag Ahil Buli | Shanta Uzir | Bihu | Successfully classified | Ahir Bhairav | 0.9448 | Bhoopali | 0.9410 | 0.0038 | Sohni | Adana | Bhimpalasi |
| 89 | BIHU_027 | Tumi Nasoni | Shanta Uzir | Bihu | Successfully classified | Ahir Bhairav | 0.9331 | Yaman Kalyan | 0.9299 | 0.0032 | Darbari | Sohni | Abhogi |
| 90 | BIHU_028 | Xunjone Nuxudhiba | Zubeen Garg, Vitali Das | Bihu | Successfully classified | Bhoopali | 0.9347 | Bahar | 0.9335 | 0.0012 | Ahir Bhairav | Madhuvanti | Malkauns |
| 91 | BIHU_029 | Morone Phulonit | Zubeen Garg, Vitali Das | Bihu | Successfully classified | Bhatiyar | 0.9391 | Darbari | 0.9380 | 0.0011 | Alhaiya Bilaval | Ahir Bhairav | Adana |
| 92 | BIHU_030 | Nila Nila Akhore | Zubeen Garg, Nirmali Das | Bihu | Successfully classified | Bilaskhani Todi | 0.9336 | Bahar | 0.9333 | 0.0004 | Adana | Puriya | Bhoopali |
| 93 | BIHU_031 | Kahili puwate | Zubeen Garg, Priyanka Bharali | Bihu | Successfully classified | Malkauns | 0.9409 | Darbari | 0.9390 | 0.0019 | suddha kalyan | Hindol | Ahir Bhairav |
| 94 | BIHU_032 | Rohare O Rohedoi | Bhitali Das | Bihu | Successfully classified | Sohni | 0.9367 | Khamaj | 0.9349 | 0.0018 | Bilaval | Abhogi | Darbari |
| 95 | BIHU_033 | Dhengkoli | Navanita Sharma | Bihu | Successfully classified | Khamaj | 0.9378 | Asavari | 0.9355 | 0.0023 | Madhuvanti | Yaman Kalyan | Bhimpalasi |
| 96 | BIHU_034 | Gabhoru Bihu | Deeplina Deka | Bihu | Successfully classified | Khamaj | 0.9449 | Bhairav | 0.9412 | 0.0037 | Bihag | Darbari | Ahir Bhairav |
| 97 | BIHU_035 | Bohagi | Rupankrita and Alankrita | Bihu | Successfully classified | Bahar | 0.9335 | Adana | 0.9314 | 0.0021 | Megh | Khamaj | Kedar |
| 98 | BIHU_036 | Podulit Bihu | Rupankrita and Alankrita | Bihu | Successfully classified | Bageshree | 0.9383 | Desh | 0.9361 | 0.0023 | Khamaj | Darbari | Puriya |
| 99 | BIHU_037 | Tok Dekhi Mor Gaa | Abhishruti Bezbaruah | Bihu | Successfully classified | Madhuvanti | 0.9381 | Alhaiya Bilaval | 0.9371 | 0.0009 | Megh | Puriya | Jaunpuri |
| 100 | DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dithi Das | Dhamail | Successfully classified | Darbari | 0.9274 | Madhuvanti | 0.9255 | 0.0019 | Poorvi | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 101 | DHAMAIL_002 | আর যাবো না জলে | Dithi Das | Dhamail | Successfully classified | Khamaj | 0.9325 | Madhuvanti | 0.9313 | 0.0012 | Poorvi | Darbari | Jaunpuri |
| 102 | DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dithi Das | Dhamail | Successfully classified | Adana | 0.9343 | Khamaj | 0.9335 | 0.0009 | Kedar | Lalit | Madhuvanti |
| 103 | DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dithi Das | Dhamail | Successfully classified | Lalit | 0.9354 | Bhatiyar | 0.9345 | 0.0008 | Bhairav | Nat Bhairav | Miyan Ki Todi orTodi or Darbari Todi |
| 104 | DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dithi Das | Dhamail | Successfully classified | Alhaiya Bilaval | 0.9257 | Nat Bhairav | 0.9242 | 0.0014 | Darbari | Miyan Ki Todi orTodi or Darbari Todi | Bageshree |
| 105 | DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Uma Das | Dhamail | Successfully classified | Purba | 0.9376 | Malkauns | 0.9354 | 0.0021 | Megh | sarang | Yaman Kalyan |
| 106 | DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Uma Das | Dhamail | Successfully classified | Abhogi | 0.9279 | sarang | 0.9258 | 0.0020 | Jaunpuri | Desh | Komal Bhairav |
| 107 | DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Uma Das | Dhamail | Successfully classified | Bahar | 0.9486 | Miyan Ki Todi orTodi or Darbari Todi | 0.9476 | 0.0010 | Purba | Bilaval | Darbari |
| 108 | DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Uma Das | Dhamail | Successfully classified | Basant Mukhari | 0.9399 | Malkauns | 0.9392 | 0.0007 | sarang | Bhairav Bahar | Miyan Ki Todi orTodi or Darbari Todi |
| 109 | DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Uma Das | Dhamail | Successfully classified | Jaunpuri | 0.9390 | Purba | 0.9356 | 0.0035 | Gujari Todi | Megh | sarang |
| 110 | DHAMAIL_011 | বল গো বল সখি | Habib Simu/Shimu | Dhamail | Successfully classified | Marva | 0.9381 | Sorath | 0.9374 | 0.0006 | Bhimpalasi | Darbari | Desh |
| 111 | DHAMAIL_012 | ইতা কিতা করে গো মনে | Habib Simu/Shimu | Dhamail | Successfully classified | Darbari | 0.9444 | Multani | 0.9407 | 0.0038 | Purba | Sorath | Madhuvanti |
| 112 | DHAMAIL_013 | শুনোগো শুনোগো সখি | Habib Simu/Shimu | Dhamail | Successfully classified | Darbari | 0.9474 | Bageshree | 0.9434 | 0.0040 | Asavari | Khamaj | N/A |
| 113 | DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Habib Simu/Shimu | Dhamail | Successfully classified | Purba | 0.9258 | Alhaiya Bilaval | 0.9233 | 0.0025 | Madhuvanti | Adana | Bhairavi |
| 114 | DHAMAIL_015 | চান্নি রাইতে আইও কালা | Habib Simu/Shimu | Dhamail | Successfully classified | Madhuvanti | 0.9452 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | 0.0070 | Multani | Nat Bhairav | Asavari |
| 115 | DHAMAIL_016 | যুগল মিলন হইলো গো | Jumiar Shanta | Dhamail | Successfully classified | Bihag | 0.9409 | Kedar | 0.9408 | 0.0001 | Multani | Bhimpalasi | Miyan Ki Todi orTodi or Darbari Todi |
| 116 | DHAMAIL_017 | বলো গো বলো সই | Pausali Banerjee | Dhamail | Successfully classified | Khamaj | 0.9351 | Bhoopali | 0.9330 | 0.0021 | Yaman Kalyan | Ahir Bhairav | Malkauns |
| 117 | DHAMAIL_018 | কত আদরে আদরে | Pausali Banerjee | Dhamail | Successfully classified | Bhupal Todi | 0.9340 | Desh | 0.9286 | 0.0053 | Yaman Kalyan | Barwa | Poorvi |
| 118 | DHAMAIL_019 | জলে গিয়েছিলাম সই | Pausali Banerjee | Dhamail | Successfully classified | Darbari | 0.9376 | Asavari | 0.9334 | 0.0042 | Adana | N/A | N/A |
| 119 | DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Pausali Banerjee | Dhamail | Successfully classified | Darbari | 0.9275 | Asavari | 0.9245 | 0.0030 | Sorath | Khamaj | Bageshree |
| 120 | DHAMAIL_021 | জলের ঘাটে | Pausali Banerjee | Dhamail | Successfully classified | Bihag | 0.9414 | Yaman Kalyan | 0.9299 | 0.0115 | Kedar | Nat Bhairav | Darbari |
| 121 | DHAMAIL_022 | Bhalo Kare Bajam Re | Maadal | Dhamail | Successfully classified | Poorvi | 0.9384 | Darbari | 0.9369 | 0.0015 | Lalit | Desh | Bahar |
| 122 | DHAMAIL_023 | Tara Lowo Go Jol | Maadal | Dhamail | Successfully classified | Khamaj | 0.9349 | Bilaskhani Todi | 0.9298 | 0.0052 | Bilaval | Kedar | Darbari |
| 123 | DHAMAIL_024 | Jhiko Jhiko Khodi Re | Maadal | Dhamail | Successfully classified | Desi | 0.9446 | Megh | 0.9420 | 0.0026 | Pahadi | Nat Bhairav | Ahir Bhairav |
| 124 | DHAMAIL_025 | Nrittangon Sunamganj | Debdas Chowdhury | Dhamail | Successfully classified | Bhairavi | 0.9243 | Malkauns | 0.9217 | 0.0025 | Madhuvanti | Sorath | Rageshree |
| 125 | DHAMAIL_026 | জলার জলার ঘাটে | Himel Dev Rahul | Dhamail | Successfully classified | Marva | 0.9157 | Ahir Bhairav | 0.9156 | 0.0002 | Gauri | Rageshree | suddha kalyan |
| 126 | DHAMAIL_027 | অচেনা এক রূপবতী | Samim Siddiki | Dhamail | Successfully classified | Ahir Bhairav | 0.9444 | Adana | 0.9435 | 0.0009 | Megh | N/A | N/A |
| 127 | DHAMAIL_028 | কুঞ্জের মাঝে | Madol Folk Band | Dhamail | Successfully classified | Megh | 0.9370 | Sorath | 0.9343 | 0.0027 | Yaman or Kalyan | N/A | N/A |
| 128 | DHAMAIL_029 | আজ রাধার কুঞ্জে | Suman Dewan | Dhamail | Successfully classified | Adana | 0.9412 | Sorath | 0.9384 | 0.0028 | Pahadi | Malkauns | N/A |
| 129 | DHAMAIL_030 | জানো নি গো সই | Jabel Raza | Dhamail | Successfully classified | Asavari | 0.9140 | Puriya | 0.9119 | 0.0021 | Adana | Sorath | Alhaiya Bilaval |
| 130 | GARBA_001 | Maa Pava Te Gadh Thi Utarya | Diwaliben Bhil | Garba | Successfully classified | Malkauns | 0.9390 | Darbari | 0.9384 | 0.0006 | Sorath | Asavari | Bhairavi |
| 131 | GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Vibha Desai, Sharavan Madhurai | Garba | Successfully classified | Ahir Bhairav | 0.9412 | Bilaskhani Todi | 0.9394 | 0.0019 | Basant Mukhari | Bhupal Todi | Bhairav Bahar |
| 132 | GARBA_003 | Ghor Andhari Re Rataldi Maa | Geeta Rabari | Garba | Successfully classified | Alhaiya Bilaval | 0.9447 | Lalit | 0.9437 | 0.0010 | Darbari | N/A | N/A |
| 133 | GARBA_004 | Meldimaa Na Kumkum Pagla | Lalita Ghodadra | Garba | Successfully classified | Megh | 0.9466 | suddha kalyan | 0.9457 | 0.0009 | Desh | Sorath | Darbari |
| 134 | GARBA_005 | Amba Aavo To Ramiye | Lalita Ghodadra | Garba | Successfully classified | Adana | 0.9489 | Yaman or Kalyan | 0.9442 | 0.0047 | Megh | N/A | N/A |
| 135 | GARBA_006 | Sonal Garbo Shire | Kishor Manraja | Garba | Successfully classified | Megh Malhar | 0.9469 | Komal Bhairav | 0.9466 | 0.0003 | Gujari Todi | Miyan Ki Todi orTodi or Darbari Todi | Malkauns |
| 136 | GARBA_007 | Maa No Garbo Re | Geeta Rabari | Garba | Successfully classified | Asavari | 0.9410 | Darbari | 0.9401 | 0.0009 | Alhaiya Bilaval | Madhuvanti | Adana |
| 137 | GARBA_008 | Unchi Talavdi Ni Kor | Asha Bhosle | Garba | Successfully classified | Asavari | 0.9397 | Marva | 0.9344 | 0.0053 | Kafi | Bahar | Hemant |
| 138 | GARBA_009 | Aavya Maa Na Norta | Pamela Jain, Divya Kumar | Garba | Successfully classified | Adana | 0.9572 | Marva | 0.9544 | 0.0027 | Rageshree | Bahar | Desh |
| 139 | GARBA_010 | Khodiyar Maa Nu Holdu | Kinjal Dave | Garba | Successfully classified | Alhaiya Bilaval | 0.9336 | Bhimpalasi | 0.9319 | 0.0017 | Darbari | Marva | Lalit |
| 140 | GARBA_011 | Sathiya Puravo | Damyanti Bardai | Garba | Successfully classified | Megh | 0.9418 | Barwa | 0.9401 | 0.0017 | Puriya | Yaman Kalyan | N/A |
| 141 | GARBA_012 | Amba Aave Aarasur Vadi | Meena Patel | Garba | Successfully classified | Yaman or Kalyan | 0.9249 | Khamaj | 0.9193 | 0.0055 | Bihag | Puriya | Megh |
| 142 | GARBA_013 | Tara Vina Maa Mane Sunu Lage | Meena Patel, Praful Dave | Garba | Successfully classified | Megh | 0.9419 | Puriya | 0.9411 | 0.0008 | Bilaskhani Todi | Bhairav Bahar | Gujari Todi |
| 143 | GARBA_014 | Garba No Rang Lagiyo | Meena Patel, Praful Dave | Garba | Successfully classified | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | Durga | 0.9379 | 0.0003 | Adana | Madhuvanti | Jaunpuri |
| 144 | GARBA_015 | Hindole Zule Amba | Meena Patel | Garba | Successfully classified | Bhoopali | 0.9291 | Barwa | 0.9284 | 0.0007 | Darbari | Bilaval | suddha kalyan |
| 145 | GARBA_016 | Jay Ambe Jagdambe Maa | Mital Gadhvi | Garba | Successfully classified | Adana | 0.9402 | Bhoopali | 0.9379 | 0.0023 | Megh | Bhairavi | Pahadi |
| 146 | GARBA_017 | Halaji Tara Hath Vakhanu | Aditya Gadhvi | Garba | Successfully classified | Darbari | 0.9499 | Asavari | 0.9451 | 0.0048 | Adana | Pahadi | N/A |
| 147 | GARBA_018 | Dholida Dhol Re Vagad | Farida Mir | Garba | Successfully classified | Bhoopali | 0.9344 | Bhimpalasi | 0.9337 | 0.0007 | Asavari | Puriya | sarang |
| 148 | GARBA_019 | Pankhida O Pankhida | Hemant Chauhan | Garba | Successfully classified | Jaunpuri | 0.9259 | Lalit | 0.9243 | 0.0016 | Desh | Durga | Adana |
| 149 | GARBA_020 | Tu To Kali Ne Kalyani | Hemant Chauhan | Garba | Successfully classified | Barwa | 0.9305 | Bhairavi | 0.9281 | 0.0024 | Yaman Kalyan | Bihag | Purba |
| 150 | GARBA_021 | He Jagjanani | Hemant Chauhan | Garba | Successfully classified | Kafi | 0.9288 | Desh | 0.9251 | 0.0038 | Yaman Kalyan | Alhaiya Bilaval | Sorath |
| 151 | GARBA_022 | Chotile Dakla Vagya | Hemant Chauhan | Garba | Successfully classified | Desh | 0.9191 | Basant Mukhari | 0.9151 | 0.0040 | Megh Malhar | Malhar | Durga |
| 152 | GARBA_023 | Jay Aadhyashakti | Hemant Chauhan | Garba | Successfully classified | Purba | 0.9348 | N/A | 0.0000 | 0.9348 | N/A | N/A | N/A |
| 153 | GARBA_024 | Dam Dam Dakla Vagya | Hemant Chauhan | Garba | Successfully classified | Bahar | 0.9234 | Gauri | 0.9228 | 0.0006 | Basant Mukhari | Khamaj | Puriya |
| 154 | GARBA_025 | Sonala Vatakadi | Praful Dave, Damyanti Bardai | Garba | Successfully classified | Marva | 0.9513 | Bilaval | 0.9485 | 0.0028 | Kedar | Bhairav | Miyan Ki Todi orTodi or Darbari Todi |
| 155 | GARBA_026 | Kaho Poonam Na Chand | Shailendra Bharti | Garba | Successfully classified | Adana | 0.9549 | Alhaiya Bilaval | 0.9547 | 0.0003 | Asavari | Malkauns | N/A |
| 156 | GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Praful Dave, Meena Patel | Garba | Successfully classified | Basant | 0.9439 | Desh | 0.9421 | 0.0018 | Jaunpuri | Madhuvanti | Bhinna Shadja |
| 157 | GARBA_028 | Lejo Rasiya Re Rumal | Arvind Barot, Vatsala Patil | Garba | Successfully classified | Marva | 0.9475 | Asavari | 0.9436 | 0.0039 | Sorath | Jaunpuri | Bhimpalasi |
| 158 | GARBA_029 | Pavana Dungre | Praful Dave | Garba | Successfully classified | Darbari | 0.9384 | Desi | 0.9379 | 0.0005 | Bihag | Kafi | Miyan Ki Todi orTodi or Darbari Todi |
| 159 | GARBA_030 | Momai Na Pare Parevado Dhume Re | Arvind Barot | Garba | Successfully classified | Khamaj | 0.9388 | Pahadi | 0.9380 | 0.0008 | Gauri | Miyan Ki Todi orTodi or Darbari Todi | Sorath |
| 160 | GARBA_031 | Rumzum Sandhani Aave Re Maa | Arvind Barot, Bhavna Rana | Garba | Successfully classified | Nat Bhairav | 0.9456 | Pahadi | 0.9453 | 0.0002 | Purba | Bihag | Khamaj |
| 161 | GHOOMAR_001 | Ghoomar | Seema Mishra | Ghoomar | Successfully classified | Khamaj | 0.9272 | Megh | 0.9263 | 0.0009 | Malkauns | Basant Mukhari | Bibhas |
| 162 | GHOOMAR_002 | Gangore/Gangaur | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9523 | Miyan Ki Todi orTodi or Darbari Todi | 0.9458 | 0.0064 | Adana | N/A | N/A |
| 163 | GHOOMAR_003 | Choumaso | Seema Mishra | Ghoomar | Successfully classified | Yaman or Kalyan | 0.9322 | Bahar | 0.9316 | 0.0006 | suddha kalyan | Purba | Bhoopali |
| 164 | GHOOMAR_004 | More Bole Re | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9425 | Adana | 0.9379 | 0.0045 | Malkauns | Basant Mukhari | Khamaj |
| 165 | GHOOMAR_005 | Podina | Seema Mishra | Ghoomar | Successfully classified | Asavari | 0.9418 | Malkauns | 0.9388 | 0.0030 | Bhairav | Purba | Marva |
| 166 | GHOOMAR_006 | Talariya Magriya | Seema Mishra | Ghoomar | Successfully classified | Adana | 0.9249 | Khamaj | 0.9217 | 0.0032 | Bhairav | Marva | Bhoopali |
| 167 | GHOOMAR_007 | Panihari Ji | Seema Mishra | Ghoomar | Successfully classified | Bhoopali | 0.9394 | Yaman or Kalyan | 0.9348 | 0.0046 | Megh | Adana | N/A |
| 168 | GHOOMAR_008 | Roomal/Rumal | Seema Mishra | Ghoomar | Successfully classified | Bilaval | 0.9330 | Yaman Kalyan | 0.9327 | 0.0004 | Darbari | Khamaj | Sohni |
| 169 | GHOOMAR_009 | Gorband | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9341 | Asavari | 0.9305 | 0.0036 | suddha kalyan | Malkauns | Ahir Bhairav |
| 170 | GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9440 | Asavari | 0.9402 | 0.0039 | Megh | Yaman or Kalyan | Malkauns |
| 171 | GHOOMAR_011 | Hivde Ro Haar | Seema Mishra | Ghoomar | Successfully classified | Barwa | 0.9375 | Khamaj | 0.9353 | 0.0021 | Ahir Bhairav | Darbari | Asavari |
| 172 | GHOOMAR_012 | Chirmi | Seema Mishra | Ghoomar | Successfully classified | Yaman Kalyan | 0.9401 | Yaman or Kalyan | 0.9378 | 0.0022 | Bhoopali | Adana | Pahadi |
| 173 | GHOOMAR_013 | Hichki (Aave Hichki) | Seema Mishra | Ghoomar | Successfully classified | Darbari | 0.9553 | Basant | 0.9499 | 0.0054 | Bahar | Adana | Bibhas |
| 174 | GHOOMAR_014 | Dal Badal | Seema Mishra | Ghoomar | Successfully classified | Adana | 0.9427 | Nat Bhairav | 0.9402 | 0.0025 | Miyan Ki Todi orTodi or Darbari Todi | Megh | Yaman or Kalyan |
| 175 | GHOOMAR_015 | Samdariyo | Seema Mishra | Ghoomar | Successfully classified | Miyan Ki Todi orTodi or Darbari Todi | 0.9456 | Bahar | 0.9454 | 0.0002 | Durga | Madhuvanti | Megh |
| 176 | GHOOMAR_016 | Jeero | Seema Mishra | Ghoomar | Successfully classified | Asavari | 0.9399 | Yaman or Kalyan | 0.9396 | 0.0003 | Adana | sarang | N/A |
| 177 | GHOOMAR_017 | Loonga Ro Batawo | Seema Mishra | Ghoomar | Successfully classified | Darbari | 0.9259 | Bhimpalasi | 0.9227 | 0.0032 | Desi | Basant Mukhari | Megh |
| 178 | GHOOMAR_018 | Taran Ri Chunadi | Seema Mishra | Ghoomar | Successfully classified | Adana | 0.9214 | Megh | 0.9207 | 0.0008 | Jaunpuri | Pahadi | N/A |
| 179 | GHOOMAR_019 | Run Jhun Baje Ghooghara | Seema Mishra | Ghoomar | Successfully classified | Alhaiya Bilaval | 0.9360 | suddha kalyan | 0.9205 | 0.0156 | Pahadi | Bhimpalasi | Adana |
| 180 | GHOOMAR_020 | Kangasiyo | Seema Mishra | Ghoomar | Successfully classified | Ahir Bhairav | 0.9296 | Desh | 0.9234 | 0.0062 | Bageshree | Bhairav | Miyan Ki Todi orTodi or Darbari Todi |
| 181 | GHOOMAR_021 | Dheere Chhal Re Panihari | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9455 | Ahir Bhairav | 0.9445 | 0.0010 | Desh | suddha kalyan | Miyan Ki Todi orTodi or Darbari Todi |
| 182 | GHOOMAR_022 | Kesariya Ro Pag | Seema Mishra | Ghoomar | Successfully classified | Asavari | 0.9419 | Khamaj | 0.9373 | 0.0046 | Sohni | Kedar | Bhoopali |
| 183 | GHOOMAR_023 | Naina Ra Lobhi | Seema Mishra | Ghoomar | Successfully classified | Miyan Ki Todi orTodi or Darbari Todi | 0.9314 | Pahadi | 0.9305 | 0.0009 | Multani | Rageshree | Madhuvanti |
| 184 | GHOOMAR_024 | Bajudan Ri Loom | Seema Mishra | Ghoomar | Successfully classified | Adana | 0.9280 | Khamaj | 0.9270 | 0.0010 | Miyan Ki Todi orTodi or Darbari Todi | Megh Malhar | Barwa |
| 185 | GHOOMAR_025 | Lehariyo | Seema Mishra | Ghoomar | Successfully classified | Darbari | 0.9203 | Bhairav | 0.9193 | 0.0010 | Kedar | Khamaj | Yaman Kalyan |
| 186 | GHOOMAR_026 | Ital Peetal | Seema Mishra | Ghoomar | Successfully classified | Durga | 0.9281 | Pahadi | 0.9254 | 0.0027 | Darbari | Alhaiya Bilaval | Miyan Ki Todi orTodi or Darbari Todi |
| 187 | GHOOMAR_027 | Moriya | Seema Mishra | Ghoomar | Successfully classified | Bhatiyar | 0.9387 | Megh | 0.9352 | 0.0036 | Lalit | Pahadi | Madhuvanti |
| 188 | GHOOMAR_028 | Cham Cham Chake Chundari | Seema Mishra | Ghoomar | Successfully classified | Gujari Todi | 0.9231 | Madhuvanti | 0.9223 | 0.0007 | Desh | Multani | Adana |
| 189 | GHOOMAR_029 | Hathelyan Re Beech | Seema Mishra | Ghoomar | Successfully classified | Barwa | 0.9361 | Khamaj | 0.9337 | 0.0023 | Megh | Bahar | Desh |
| 190 | GHOOMAR_030 | Dar Lanjo | Seema Mishra | Ghoomar | Successfully classified | Bhairav | 0.9436 | Asavari | 0.9419 | 0.0017 | Bhoopali | Adana | Megh Malhar |
| 191 | JHUMUR_001 | Kati Khane Bole | Dulal Manki | Jhumur | Successfully classified | Marva | 0.9283 | Bhatiyar | 0.9261 | 0.0022 | Madhuvanti | Bhinna Shadja | Malkauns |
| 192 | JHUMUR_0010 | Ekadoshi Ratiya | Zubeen Garg | Jhumur | Successfully classified | Lalit | 0.9387 | Bilaval | 0.9379 | 0.0008 | Megh | Barwa | Madhuvanti |
| 193 | JHUMUR_0011 | Barire Simare | Zubeen Garg | Jhumur | Successfully classified | Ahir Bhairav | 0.9385 | Darbari | 0.9343 | 0.0041 | Pahadi | suddha kalyan | N/A |
| 194 | JHUMUR_0012 | Mor Dhoni Go | Zubeen Garg | Jhumur | Successfully classified | Malkauns | 0.9293 | Madhuvanti | 0.9192 | 0.0101 | Bilaval | Asavari | Sorath |
| 195 | JHUMUR_0013 | Jhuli Jhule | Zubeen Garg | Jhumur | Successfully classified | Desi | 0.9251 | Madhuvanti | 0.9251 | 0.0000 | Khamaj | Marva | Ahir Bhairav |
| 196 | JHUMUR_0014 | Kansiya | Amrit Tanti | Jhumur | Successfully classified | Bageshree | 0.9427 | Kedar | 0.9418 | 0.0009 | Asavari | Yaman Kalyan | Darbari |
| 197 | JHUMUR_0015 | Jhatka | Amrit Tanti | Jhumur | Successfully classified | Bhoopali | 0.9381 | Sorath | 0.9366 | 0.0015 | Megh | Barwa | Adana |
| 198 | JHUMUR_0016 | Gohona | Amrit Tanti | Jhumur | Successfully classified | Bageshree | 0.9419 | Khamaj | 0.9387 | 0.0032 | Darbari | Sohni | Asavari |
| 199 | JHUMUR_0017 | Hariya | Amrit Tanti | Jhumur | Successfully classified | Kedar | 0.9342 | Darbari | 0.9303 | 0.0039 | Bhoopali | Asavari | N/A |
| 200 | JHUMUR_0018 | Koromoti | Amrit Tanti | Jhumur | Successfully classified | Khamaj | 0.9395 | Bageshree | 0.9390 | 0.0005 | Kedar | Adana | Bilaval |
| 201 | JHUMUR_0019 | Sonamoni | Amrit Tanti | Jhumur | Successfully classified | Adana | 0.9368 | Desh | 0.9367 | 0.0002 | Sorath | Bahar | Barwa |
| 202 | JHUMUR_002 | Nistei | Dulal Manki | Jhumur | Successfully classified | Bageshree | 0.9418 | Sohni | 0.9418 | 0.0000 | Pahadi | Marva | Sorath |
| 203 | JHUMUR_0020 | Chaliya Chhodi | Amrit Tanti | Jhumur | Successfully classified | Bhoopali | 0.9277 | Malkauns | 0.9247 | 0.0030 | Khamaj | N/A | N/A |
| 204 | JHUMUR_0021 | Prem Nadi Tire Tire | Shaan Tanti | Jhumur | Successfully classified | Bhairavi | 0.9313 | Ahir Bhairav | 0.9311 | 0.0002 | Adana | Sorath | Megh Malhar |
| 205 | JHUMUR_0022 | Deewana | Shaan Tanti | Jhumur | Successfully classified | Marva | 0.9413 | Bhimpalasi | 0.9378 | 0.0035 | Sorath | Desh | Adana |
| 206 | JHUMUR_0023 | Koliye Juge Alai Doyal | Navin Tanti | Jhumur | Successfully classified | Lalit | 0.9528 | Megh | 0.9527 | 0.0000 | Adana | N/A | N/A |
| 207 | JHUMUR_0024 | Laal Pari | Bejnath Tanti | Jhumur | Successfully classified | Madhuvanti | 0.9358 | Sorath | 0.9358 | 0.0000 | Desh | Megh | Adana |
| 208 | JHUMUR_0025 | Tore Bina | Bejnath Tanti | Jhumur | Successfully classified | Asavari | 0.9524 | Adana | 0.9521 | 0.0003 | Bhoopali | Alhaiya Bilaval | N/A |
| 209 | JHUMUR_0026 | Koromake Akhora | Bejnath Tanti | Jhumur | Successfully classified | Bilaval | 0.9437 | Darbari | 0.9414 | 0.0023 | Lalit | Bhimpalasi | Khamaj |
| 210 | JHUMUR_0027 | The Jhumur Song | Neel Akash | Jhumur | Successfully classified | Nat Bhairav | 0.9461 | Asavari | 0.9452 | 0.0009 | Bhairav | Adana | Megh |
| 211 | JHUMUR_0028 | Genda Phool | Neel Akash, Jyotika Baruha | Jhumur | Successfully classified | Bageshree | 0.9461 | Kedar | 0.9430 | 0.0030 | Adana | Darbari | Bilaval |
| 212 | JHUMUR_0029 | Ja Ja Re Mini | Neel Akash | Jhumur | Successfully classified | Ahir Bhairav | 0.9344 | Asavari | 0.9334 | 0.0010 | Pahadi | Miyan Ki Todi orTodi or Darbari Todi | Darbari |
| 213 | JHUMUR_003 | Kuilu Kuilu | Dulal Manki | Jhumur | Successfully classified | Darbari | 0.9368 | Bhinna Shadja | 0.9347 | 0.0021 | Bhairav | Basant | Basant Mukhari |
| 214 | JHUMUR_0030 | Goya | Neel Akash | Jhumur | Successfully classified | Adana | 0.9377 | Asavari | 0.9369 | 0.0008 | N/A | N/A | N/A |
| 215 | JHUMUR_004 | Sajanigo | Dulal Manki | Jhumur | Successfully classified | Desi | 0.9391 | Yaman or Kalyan | 0.9328 | 0.0063 | Rageshree | Adana | Basant |
| 216 | JHUMUR_005 | Sol Sokhi | Dulal Manki | Jhumur | Successfully classified | Deepak | 0.9357 | Bhatiyar | 0.9334 | 0.0023 | Marva | Lalit | Bhairav |
| 217 | JHUMUR_006 | Aji Ei Suprabhate | Dulal Manki | Jhumur | Successfully classified | Bilaval | 0.9240 | Bhairavi | 0.9224 | 0.0016 | Kafi | Sorath | Marva |
| 218 | JHUMUR_007 | Kale Gori Muke | Dulal Manki | Jhumur | Successfully classified | Nat Bhairav | 0.9320 | Malhar | 0.9309 | 0.0011 | Malkauns | Basant | Basant Mukhari |
| 219 | JHUMUR_008 | Tohe Gori Nawa | Dulal Manki | Jhumur | Successfully classified | Bahar | 0.9298 | Komal Bhairav | 0.9280 | 0.0018 | Sohni | Malkauns | Bhinna Shadja |
| 220 | JHUMUR_009 | Moi to Jabo Kam Jari | Zubeen Garg | Jhumur | Successfully classified | Adana | 0.9310 | sarang | 0.9304 | 0.0007 | Miyan Ki Todi orTodi or Darbari Todi | Darbari | Sohni |
| 221 | LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Sulochana Chavan | Lavani | Successfully classified | Desh | 0.9381 | Bilaskhani Todi | 0.9374 | 0.0007 | Basant | Puriya | Megh |
| 222 | LAVANI_002 | Padarawarti Jartaricha | Sulochana Chavan | Lavani | Successfully classified | Marva | 0.9466 | Megh Malhar | 0.9454 | 0.0012 | Asavari | Kedar | Nat Bhairav |
| 223 | LAVANI_003 | Solava Varis Dhokyacha | Sulochana Chavan | Lavani | Successfully classified | Bahar | 0.9312 | Bhairavi | 0.9306 | 0.0006 | Adana | Komal Bhairav | Asavari |
| 224 | LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Sulochana Chavan | Lavani | Successfully classified | Bahar | 0.9427 | Bihag | 0.9410 | 0.0017 | Khamaj | Bageshree | Bhimpalasi |
| 225 | LAVANI_005 | Naav Gaav Kashala Pustay | Sulochana Chavan | Lavani | Successfully classified | Nat Bhairav | 0.9527 | Rageshree | 0.9500 | 0.0027 | Megh Malhar | Malhar | N/A |
| 226 | LAVANI_006 | Kalidar Kapuri Paan | Sulochana Chavan | Lavani | Successfully classified | Bahar | 0.9247 | Nat Bhairav | 0.9216 | 0.0031 | Kafi | Basant | suddha kalyan |
| 227 | LAVANI_007 | Mala London Foreign Dakhwa | Sulochana Chavan | Lavani | Successfully classified | Malkauns | 0.9369 | Bahar | 0.9315 | 0.0053 | Yaman Kalyan | Bilaval | Yaman or Kalyan |
| 228 | LAVANI_008 | Disla Ga Bai Disla | Usha Mangeshkar | Lavani | Successfully classified | Kedar | 0.9498 | Asavari | 0.9493 | 0.0005 | Gauri | Marva | Megh Malhar |
| 229 | LAVANI_009 | Chabidar Chabi | Usha Mangeshkar | Lavani | Successfully classified | suddha kalyan | 0.9305 | Yaman or Kalyan | 0.9276 | 0.0029 | Puriya | Pahadi | Desh |
| 230 | LAVANI_010 | Raya Chala Ghodyavarti Basu | Usha Mangeshkar | Lavani | Successfully classified | Asavari | 0.9530 | Miyan Ki Todi orTodi or Darbari Todi | 0.9494 | 0.0036 | Marva | Bihag | Lalit |
| 231 | LAVANI_011 | Pikala Jambhul Todu Naka | Usha Mangeshkar | Lavani | Successfully classified | Asavari | 0.9584 | Miyan Ki Todi orTodi or Darbari Todi | 0.9557 | 0.0026 | Bageshree | Rageshree | Puriya |
| 232 | LAVANI_012 | Naki Doli Chhaan | Usha Mangeshkar | Lavani | Successfully classified | Puriya | 0.9578 | Kedar | 0.9494 | 0.0084 | Bageshree | Megh | Bihag |
| 233 | LAVANI_013 | Aai Majhya Lagnachi | Usha Mangeshkar | Lavani | Successfully classified | Malhar | 0.9455 | Marva | 0.9439 | 0.0017 | Bhoopali | Durga | N/A |
| 234 | LAVANI_014 | Wajle Ki Bara | Bela Shende | Lavani | Successfully classified | Bahar | 0.9337 | Bhoopali | 0.9333 | 0.0004 | Sohni | Puriya | Bhairavi |
| 235 | LAVANI_015 | Apsara Aali | Bela Shende | Lavani | Successfully classified | Basant | 0.9305 | Sorath | 0.9297 | 0.0008 | Bhimpalasi | Marva | Bhinna Shadja |
| 236 | LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Bela Shende | Lavani | Successfully classified | Shree | 0.9252 | Bahar | 0.9245 | 0.0007 | Bhimpalasi | Sohni | Adana |
| 237 | LAVANI_017 | Kashyala Lavato | Bela Shende | Lavani | Successfully classified | Barwa | 0.9223 | Darbari | 0.9137 | 0.0086 | Alhaiya Bilaval | Ahir Bhairav | Kafi |
| 238 | LAVANI_018 | Chhatishi Kavlun Dhara | Surekha Punekar | Lavani | Successfully classified | Bahar | 0.9357 | Madhuvanti | 0.9350 | 0.0007 | Puriya | Khamaj | Darbari |
| 239 | LAVANI_019 | Jhampar Majha Dheel | Surekha Punekar | Lavani | Successfully classified | Yaman Kalyan | 0.9265 | Miyan Ki Todi orTodi or Darbari Todi | 0.9250 | 0.0015 | Puriya | Malkauns | Desi |
| 240 | LAVANI_020 | Themba Thembana Kelai | Surekha Punekar | Lavani | Successfully classified | suddha kalyan | 0.9305 | Yaman or Kalyan | 0.9276 | 0.0029 | Puriya | Pahadi | Desh |
| 241 | LAVANI_021 | Sola Hazarat | Uttara Kelkar | Lavani | Successfully classified | Marva | 0.9375 | Bilaskhani Todi | 0.9350 | 0.0025 | Desh | Bageshree | Puriya |
| 242 | LAVANI_022 | Hi Natrangi Naar | Ravindra Sathe | Lavani | Successfully classified | Puriya | 0.9424 | Marva | 0.9405 | 0.0019 | Bageshree | Bhairav | Bhairavi |
| 243 | LAVANI_023 | Saaman Avjad Majha | Anand Shinde | Lavani | Successfully classified | Yaman Kalyan | 0.9265 | Miyan Ki Todi orTodi or Darbari Todi | 0.9219 | 0.0046 | Bageshree | Malkauns | Desh |
| 244 | LAVANI_024 | Majha Taat Kar Aadhi | Anand Shinde | Lavani | Successfully classified | suddha kalyan | 0.9290 | Yaman or Kalyan | 0.9283 | 0.0008 | Puriya | Pahadi | Desh |
| 245 | LAVANI_025 | Bai Mi Patang Udvit Hote | Asha Bhosle | Lavani | Successfully classified | Yaman or Kalyan | 0.9356 | Yaman Kalyan | 0.9351 | 0.0006 | Gauri | Durga | Bahar |
| 246 | LAVANI_026 | Bai Mazi Karangali Modali | Asha Bhosle | Lavani | Successfully classified | Basant Mukhari | 0.9551 | Puriya | 0.9550 | 0.0001 | Bihag | Rageshree | Darbari |
| 247 | LAVANI_027 | Raya Mala Pavsat Neu Naka | Pushpa Pagdhare | Lavani | Successfully classified | Marva | 0.9522 | Bageshree | 0.9471 | 0.0051 | Asavari | Purba | Bhoopali |
| 248 | LAVANI_028 | Navi Navi Navari Mee | Pushpa Pagdhare | Lavani | Successfully classified | Malhar | 0.9410 | Khamaj | 0.9365 | 0.0045 | Desh | Marva | Kedar |
| 249 | LAVANI_029 | Ugvali Shukrachi Chandni | Aarti Anklikar | Lavani | Successfully classified | Ahir Bhairav | 0.9206 | Desh | 0.9204 | 0.0003 | Puriya | Darbari | Barwa |
| 250 | LAVANI_030 | Ghotala Ghotala Zala | Vaishali Samant | Lavani | Successfully classified | Bilaskhani Todi | 0.9362 | Marva | 0.9360 | 0.0001 | Desh | Bageshree | Puriya |
| 251 | LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Urmila Dhangkar | Lavani | Successfully classified | Adana | 0.9340 | Bhoopali | 0.9327 | 0.0014 | Yaman or Kalyan | N/A | N/A |
| 252 | LAVANI_032 | Aase Wajwa Ki | Bela Shende and chorus | Lavani | Successfully classified | Megh | 0.9426 | Miyan Ki Todi orTodi or Darbari Todi | 0.9421 | 0.0005 | Nat Bhairav | Puriya | Desh |
| 253 | LAVANI_033 | Dolyavar Gogal Lava | Various Artists | Lavani | Successfully classified | Darbari | 0.9348 | Miyan Ki Todi orTodi or Darbari Todi | 0.9345 | 0.0002 | Purba | sarang | Kedar |
| 254 | LAVANI_034 | Reshmachya Reghani | Asha Bhosle | Lavani | Successfully classified | Khamaj | 0.9433 | Gauri | 0.9429 | 0.0004 | Basant | Bahar | Malkauns |
| 255 | LAVANI_035 | Chala Jejurila Jau | Uttara Kelkar | Lavani | Successfully classified | Yaman Kalyan | 0.9391 | Desh | 0.9385 | 0.0006 | Alhaiya Bilaval | Darbari | Bhairavi |
| 256 | MAAND_001 | Kesariya Balam Aavo Ni | Allah Jilai Bai | Maand | Successfully classified | Darbari | 0.9456 | Bahar | 0.9423 | 0.0034 | Purba | Multani | Jaunpuri |
| 257 | MAAND_002 | Moray Aaj Aaye Saiyan | Allah Jilai Bai | Maand | Successfully classified | Hemant | 0.9179 | Bihag | 0.9174 | 0.0005 | Bibhas | Lalit | Desh |
| 258 | MAAND_003 | Moomal | Allah Jilai Bai | Maand | Successfully classified | Madhuvanti | 0.9465 | Jaunpuri | 0.9453 | 0.0012 | Multani | Desh | Purba |
| 259 | MAAND_004 | Ghoomar | Allah Jilai Bai | Maand | Successfully classified | Purba | 0.9283 | Bilaskhani Todi | 0.9236 | 0.0047 | N/A | N/A | N/A |
| 260 | MAAND_005 | Bai Sa Ra Beera | Allah Jilai Bai | Maand | Successfully classified | Megh | 0.9415 | Bahar | 0.9408 | 0.0008 | Miyan Ki Todi orTodi or Darbari Todi | Purba | Bhairav Bahar |
| 261 | MAAND_006 | Jawai Balha Lagoji | Allah Jilai Bai | Maand | Successfully classified | Bahar | 0.9220 | Bageshree | 0.9212 | 0.0007 | Asavari | Gauri | Yaman or Kalyan |
| 262 | MAAND_007 | Kesariya Balam Aavo Ni | Gavri Devi | Maand | Successfully classified | Darbari | 0.9233 | Madhuvanti | 0.9231 | 0.0002 | Bahar | Bilaskhani Todi | Sorath |
| 263 | MAAND_008 | Kesariya Balam Awo Ni Padharo | Seema Mishra | Maand | Successfully classified | Bhatiyar | 0.9289 | Miyan Ki Todi orTodi or Darbari Todi | 0.9287 | 0.0002 | Madhuvanti | Poorvi | Hemant |
| 264 | MAAND_009 | Kesariya Ro Pag (Ghoomar) | Seema Mishra | Maand | Successfully classified | Khamaj | 0.9444 | Ahir Bhairav | 0.9431 | 0.0013 | Bhoopali | Asavari | Sohni |
| 265 | MAAND_010 | Badila Yahi Rahiyo Saa | Banarsi Babu | Maand | Successfully classified | Gauri | 0.9537 | Bageshree | 0.9457 | 0.0080 | Adana | Desh | Khamaj |
| 266 | MAAND_011 | Taniya Resham Ki | Banarsi Babu | Maand | Successfully classified | Adana | 0.9454 | Khamaj | 0.9430 | 0.0024 | Alhaiya Bilaval | N/A | N/A |
| 267 | MAAND_012 | Kad Janulee Sasre | Rehana Mirza | Maand | Successfully classified | Alhaiya Bilaval | 0.9421 | Bhairavi | 0.9388 | 0.0034 | Darbari | Bhairav | Bahar |
| 268 | MAAND_013 | Le Dekh Nangi Melo | Rehana Mirza | Maand | Successfully classified | Khamaj | 0.9477 | Malhar | 0.9466 | 0.0011 | Megh Malhar | Purba | Yaman Kalyan |
| 269 | MAAND_014 | The To Piyo Sadarudi | Rehana Mirza | Maand | Successfully classified | Rageshree | 0.9392 | Abhogi | 0.9385 | 0.0007 | Bilaskhani Todi | Marva | Durga |
| 270 | MAAND_015 | Kesariya Balam (Desert Slide) | Anwar Khan | Maand | Successfully classified | Miyan Ki Todi orTodi or Darbari Todi | 0.9267 | Malkauns | 0.9239 | 0.0029 | Darbari | Sohni | Madhuvanti |
| 271 | MAAND_016 | Kesariya Balam (DIFF 2015) | Anwar Khan | Maand | Successfully classified | Bhairavi | 0.9186 | Madhuvanti | 0.9147 | 0.0039 | Pahadi | Purba | Hemant |
| 272 | MAAND_017 | Apne Desh Ri Aajadi | Anwar Khan | Maand | Successfully classified | Sorath | 0.9351 | Basant | 0.9312 | 0.0038 | Adana | N/A | N/A |
| 273 | MAAND_018 | Kesariya Balam Aao Ni | Anwar Khan | Maand | Successfully classified | Ahir Bhairav | 0.9354 | Gauri | 0.9319 | 0.0035 | Sohni | Khamaj | Basant |
| 274 | MAAND_019 | Raag Maand Bandish | Mame Khan | Maand | Successfully classified | Bilaval | 0.9388 | Purba | 0.9388 | 0.0000 | Sorath | Khamaj | Adana |
| 275 | MAAND_020 | Rajasthani Folk Raag Maand | Mame Khan | Maand | Successfully classified | Bilaval | 0.9409 | Purba | 0.9397 | 0.0012 | Sorath | Khamaj | suddha kalyan |
| 276 | MAAND_021 | Chaudhary | Mame Khan | Maand | Successfully classified | Ahir Bhairav | 0.9324 | Desh | 0.9307 | 0.0017 | Nat Bhairav | Bhairavi | suddha kalyan |
| 277 | MAAND_022 | Chhap Tilak Sab Cheeni | Kirti Sagathia | Maand | Successfully classified | Adana | 0.9323 | Bahar | 0.9323 | 0.0001 | Darbari | Nat Bhairav | suddha kalyan |
| 278 | MAAND_023 | Balam | Swaroop Khan | Maand | Successfully classified | Darbari | 0.9495 | Asavari | 0.9451 | 0.0044 | N/A | N/A | N/A |
| 279 | MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Swaroop Khan Sura | Maand | Successfully classified | Marva | 0.9147 | Ahir Bhairav | 0.9136 | 0.0011 | Barwa | Bhairavi | Bibhas |
| 280 | MAAND_025 | Kurja Mhari Bhanwar Milade | Sikander Khan Baru | Maand | Successfully classified | Khamaj | 0.9313 | Adana | 0.9296 | 0.0017 | Bahar | Pahadi | Miyan Ki Todi orTodi or Darbari Todi |
| 281 | MAAND_026 | Darude Ro Rog Lagiyo | Sikander Khan Baru | Maand | Successfully classified | Ahir Bhairav | 0.9131 | Puriya | 0.9098 | 0.0033 | Darbari | Asavari | N/A |
| 282 | MAAND_027 | Kesariya Balam Padharo | Bhungar Khan | Maand | Successfully classified | Komal Bhairav | 0.9439 | Bhoopali | 0.9401 | 0.0038 | Bhimpalasi | Khamaj | Purba |
| 283 | MAAND_028 | Mohe Aai Na Jag Ki Laaj | Kheta Khan, Fakira Khan Bhadresh | Maand | Successfully classified | Kafi | 0.9416 | Pahadi | 0.9397 | 0.0019 | Desh | Bilaskhani Todi | Malhar |
| 284 | MAAND_029 | Kalali | Latif Khan | Maand | Successfully classified | Malkauns | 0.9186 | Bhairavi | 0.9169 | 0.0017 | Megh | Desh | Ahir Bhairav |
| 285 | MAAND_030 | Kesariya Balam | Raitila (Group) | Maand | Successfully classified | Yaman Kalyan | 0.9278 | Sorath | 0.9263 | 0.0014 | Purba | Malkauns | Puriya |
| 286 | MAAND_031 | Joori Thaane Na Maare Raad | Jameela Bai & Kulsum Bai | Maand | Successfully classified | Yaman Kalyan | 0.9218 | Bhimpalasi | 0.9212 | 0.0007 | Bilaskhani Todi | Lalit | Basant |
| 287 | MAAND_032 | Alija Dhola Ekladi | Jameela Bai & Kulsum Bai | Maand | Successfully classified | Desh | 0.9228 | Bhimpalasi | 0.9190 | 0.0037 | Yaman Kalyan | Bhairavi | N/A |
| 288 | MAAND_033 | Mumal | Jameela Bai & Kulsum Bai | Maand | Successfully classified | Purba | 0.9342 | Yaman Kalyan | 0.9331 | 0.0012 | Bhimpalasi | Bhatiyar | Bilaskhani Todi |
| 289 | MAAND_034 | Mhari Sajni Ae | Nand Lal Bhat | Maand | Successfully classified | Sorath | 0.9439 | Bhairavi | 0.9420 | 0.0019 | Yaman Kalyan | Puriya | Darbari |
| 290 | MAAND_035 | Rajwadi Mehfil | Nand Lal Bhat | Maand | Successfully classified | Alhaiya Bilaval | 0.9365 | Marva | 0.9363 | 0.0002 | Megh | Bhairavi | Basant |
| 291 | MAAND_036 | Thane Samjhaye Sakha Sa | Munshi Khan | Maand | Successfully classified | Darbari | 0.9420 | Puriya | 0.9395 | 0.0025 | Bibhas | Adana | Basant |
| 292 | MAAND_037 | Padharo Mhare Mehal | Munshi Khan | Maand | Successfully classified | Bhupal Todi | 0.9445 | Ramkali | 0.9417 | 0.0027 | Komal Bhairav | Bibhas | Adana |
| 293 | MAAND_038 | Daru Dakha Ro | Munshi Khan | Maand | Successfully classified | Gauri | 0.9424 | Alhaiya Bilaval | 0.9399 | 0.0025 | Malhar | Khamaj | Kafi |
| 294 | MAAND_039 | Mahara Sajan Base | Munshi Khan | Maand | Successfully classified | Purba | 0.9428 | Desh | 0.9375 | 0.0053 | Yaman Kalyan | Adana | Bhatiyar |
| 295 | SAMBALPURI_001 | Nabu Ki Sambalpur | Human Sagar | Sambalpuri | Successfully classified | Ahir Bhairav | 0.9407 | Malkauns | 0.9363 | 0.0044 | Pahadi | Bhimpalasi | Desh |
| 296 | SAMBALPURI_002 | Tor Lagi Marijibi | Human Sagar | Sambalpuri | Successfully classified | Bilaval | 0.9375 | Ahir Bhairav | 0.9366 | 0.0008 | Sorath | Darbari | Desh |
| 297 | SAMBALPURI_003 | Sacha Lover | Human Sagar | Sambalpuri | Successfully classified | Basant | 0.9391 | Darbari | 0.9370 | 0.0021 | Asavari | Sorath | Marva |
| 298 | SAMBALPURI_004 | Pardesia Raja | Human Sagar | Sambalpuri | Successfully classified | Sorath | 0.9508 | Darbari | 0.9491 | 0.0017 | Bageshree | Desh | Miyan Ki Todi orTodi or Darbari Todi |
| 299 | SAMBALPURI_005 | Lal Sadhi | Human Sagar | Sambalpuri | Successfully classified | Bilaval | 0.9279 | Desh | 0.9254 | 0.0025 | Bageshree | Bhairavi | Kedar |
| 300 | SAMBALPURI_006 | Rasia | Mantu Chhuria | Sambalpuri | Successfully classified | Bilaval | 0.9351 | Bageshree | 0.9276 | 0.0074 | Bhimpalasi | Asavari | Desh |
| 301 | SAMBALPURI_007 | A Mo Dil | Mantu Chhuria | Sambalpuri | Successfully classified | Darbari | 0.9364 | Bageshree | 0.9330 | 0.0034 | Adana | Asavari | Malkauns |
| 302 | SAMBALPURI_008 | Hae Go | Mantu Chhuria | Sambalpuri | Successfully classified | Darbari | 0.9347 | Megh | 0.9345 | 0.0002 | Sorath | Bilaval | Ahir Bhairav |
| 303 | SAMBALPURI_009 | Aukaat | Mantu Chhuria | Sambalpuri | Successfully classified | Megh | 0.9504 | Bhatiyar | 0.9495 | 0.0008 | Sohni | Bhimpalasi | Asavari |
| 304 | SAMBALPURI_010 | Rani Guri | Mantu Chhuria | Sambalpuri | Successfully classified | Bageshree | 0.9397 | Puriya | 0.9367 | 0.0030 | Alhaiya Bilaval | Adana | Bhoopali |
| 305 | SAMBALPURI_011 | Gidja Gija Re | Asima Panda | Sambalpuri | Successfully classified | Khamaj | 0.9379 | Bageshree | 0.9376 | 0.0003 | Bilaval | Adana | Abhogi |
| 306 | SAMBALPURI_012 | Tui Bewafa | Asima Panda | Sambalpuri | Successfully classified | Desh | 0.9439 | Adana | 0.9402 | 0.0037 | Miyan Ki Todi orTodi or Darbari Todi | Sorath | Ahir Bhairav |
| 307 | SAMBALPURI_013 | Ramagaru | Asima Panda | Sambalpuri | Successfully classified | Sorath | 0.9276 | Desh | 0.9275 | 0.0001 | Alhaiya Bilaval | N/A | N/A |
| 308 | SAMBALPURI_014 | Deewana Banalu Rani | Asima Panda | Sambalpuri | Successfully classified | Bageshree | 0.9338 | Megh | 0.9335 | 0.0002 | Rageshree | Ahir Bhairav | Adana |
| 309 | SAMBALPURI_015 | SambalPuri Saiyaan | Asima Panda | Sambalpuri | Successfully classified | Sorath | 0.9390 | Desh | 0.9374 | 0.0016 | Adana | Miyan Ki Todi orTodi or Darbari Todi | Ahir Bhairav |
| 310 | SAMBALPURI_016 | Anta Halei Halei | Ruku Suna | Sambalpuri | Successfully classified | Alhaiya Bilaval | 0.9476 | Bhairavi | 0.9467 | 0.0009 | Pahadi | Darbari | Ahir Bhairav |
| 311 | SAMBALPURI_017 | Chandir Paenjhal | Ruku Suna | Sambalpuri | Successfully classified | Adana | 0.9446 | Asavari | 0.9443 | 0.0003 | Desh | Alhaiya Bilaval | Darbari |
| 312 | SAMBALPURI_018 | Prem Paga | Ruku Suna | Sambalpuri | Successfully classified | Darbari | 0.9492 | Asavari | 0.9469 | 0.0023 | Adana | Ahir Bhairav | Pahadi |
| 313 | SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Ruku Suna | Sambalpuri | Successfully classified | Megh | 0.9427 | Hindol | 0.9419 | 0.0007 | Alhaiya Bilaval | Adana | Nat Bhairav |
| 314 | SAMBALPURI_020 | A Labanga Lata | Ruku Suna | Sambalpuri | Successfully classified | Bhairavi | 0.9297 | Bhoopali | 0.9291 | 0.0006 | Asavari | Abhogi | Pahadi |
| 315 | SAMBALPURI_021 | Badnam | Prakash Jal | Sambalpuri | Successfully classified | Rageshree | 0.9225 | Kedar | 0.9185 | 0.0040 | Pahadi | Ahir Bhairav | Yaman or Kalyan |
| 316 | SAMBALPURI_022 | Dil Bhitre Lekhideli | Prakash Jal | Sambalpuri | Successfully classified | Pahadi | 0.9342 | Ahir Bhairav | 0.9319 | 0.0023 | N/A | N/A | N/A |
| 317 | SAMBALPURI_023 | Bewafa Mita | Prakash Jal | Sambalpuri | Successfully classified | Adana | 0.9353 | Durga | 0.9340 | 0.0013 | Bhoopali | Pahadi | Marva |
| 318 | SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Prakash Jal | Sambalpuri | Successfully classified | Pahadi | 0.9374 | Nat Bhairav | 0.9368 | 0.0006 | Miyan Ki Todi orTodi or Darbari Todi | Bihag | Ahir Bhairav |
| 319 | SAMBALPURI_025 | Le Le Baula | Antara Chakrabarty | Sambalpuri | Successfully classified | Asavari | 0.9484 | Adana | 0.9467 | 0.0017 | Alhaiya Bilaval | Puriya | Ahir Bhairav |
| 320 | SAMBALPURI_026 | Chahanewali | Antara Chakrabarty | Sambalpuri | Excluded | NO PREDICTION | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 321 | SAMBALPURI_027 | Feel My Love | Antara Chakrabarty | Sambalpuri | Successfully classified | Marva | 0.9430 | Madhuvanti | 0.9425 | 0.0005 | Bhimpalasi | Desh | Darbari |
| 322 | SAMBALPURI_028 | Kulfirani Chocobar | Ira Mohanty | Sambalpuri | Successfully classified | Marva | 0.9496 | Megh | 0.9429 | 0.0067 | Asavari | Alhaiya Bilaval | N/A |
| 323 | SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Ira Mohanty | Sambalpuri | Successfully classified | Khamaj | 0.9419 | Bageshree | 0.9366 | 0.0053 | Asavari | Adana | Darbari |
| 324 | SAMBALPURI_030 | Sundri Janha | Ira Mohanty | Sambalpuri | Successfully classified | Bageshree | 0.9371 | Asavari | 0.9356 | 0.0015 | Ahir Bhairav | Kedar | Darbari |
| 325 | SAMBALPURI_031 | Rangabati | Jitendra Haripal | Sambalpuri | Successfully classified | Bahar | 0.9151 | Kedar | 0.9113 | 0.0038 | Adana | Rageshree | Bhairav |
| 326 | SAMBALPURI_032 | Aha Mor Galar Malire | Jitendra Haripal | Sambalpuri | Successfully classified | Basant | 0.9320 | Marva | 0.9267 | 0.0052 | Ahir Bhairav | Sorath | Malkauns |
| 327 | SAMBALPURI_033 | Din Gani Gani | Jitendra Haripal | Sambalpuri | Successfully classified | Shree | 0.9340 | Alhaiya Bilaval | 0.9339 | 0.0001 | Khamaj | Desi | Asavari |
| 328 | SAMBALPURI_034 | Namo Namo Ago | Jitendra Haripal | Sambalpuri | Successfully classified | Bhairav | 0.9314 | Kedar | 0.9306 | 0.0009 | Rageshree | Bhoopali | Puriya |

## PART 2 — DETAILED SONG-BY-SONG SECTIONS

## 001 — Agun Pani
**Original Song ID:** BAUL_001

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Madhuvanti | 0.9390 |
| 2 | Darbari | 0.9383 |
| 3 | Bhairavi | N/A |
| 4 | Purba | N/A |
| 5 | Bahar | N/A |

**Top-1 Candidate:** Madhuvanti

**Top-1 Similarity:** 0.9390

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Madhuvanti was the highest-ranked candidate in the TRF reference space for this recording.

## 002 — Ja Lolite Anga Kushum Tule
**Original Song ID:** BAUL_002

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Abhogi | 0.9225 |
| 2 | Malkauns | 0.9201 |
| 3 | Darbari | N/A |
| 4 | Yaman Kalyan | N/A |
| 5 | Alhaiya Bilaval | N/A |

**Top-1 Candidate:** Abhogi

**Top-1 Similarity:** 0.9225

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Abhogi was the highest-ranked candidate in the TRF reference space for this recording.

## 003 — Kalkatou Mon Heshekeley
**Original Song ID:** BAUL_003

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9422 |
| 2 | Malkauns | 0.9393 |
| 3 | Abhogi | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9422

**Top-1 vs Top-2 Margin:** 0.0029

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 004 — Ananda Lahari
**Original Song ID:** BAUL_004

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9303 |
| 2 | Bhairavi | 0.9264 |
| 3 | Khamaj | N/A |
| 4 | Malhar | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9303

**Top-1 vs Top-2 Margin:** 0.0039

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 005 — Amar Aei Rudir Dhara
**Original Song ID:** BAUL_005

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9324 |
| 2 | Alhaiya Bilaval | 0.9293 |
| 3 | Malkauns | N/A |
| 4 | Purba | N/A |
| 5 | sarang | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9324

**Top-1 vs Top-2 Margin:** 0.0031

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 006 — Hari Balo Re
**Original Song ID:** BAUL_006

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9325 |
| 2 | Malkauns | 0.9291 |
| 3 | Puriya | N/A |
| 4 | Darbari | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9325

**Top-1 vs Top-2 Margin:** 0.0033

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 007 — Gour Lila Bazare
**Original Song ID:** BAUL_007

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9373 |
| 2 | Megh | 0.9314 |
| 3 | Darbari | N/A |
| 4 | Yaman or Kalyan | N/A |
| 5 | Kafi | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9373

**Top-1 vs Top-2 Margin:** 0.0059

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 008 — Tripti
**Original Song ID:** BAUL_008

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.8812 |
| 2 | Sorath | 0.8810 |
| 3 | Basant | N/A |
| 4 | Bhairavi | N/A |
| 5 | Malhar | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.8812

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 009 — Kon Ba Zharir
**Original Song ID:** BAUL_009

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Barwa | 0.9362 |
| 2 | Lalit | 0.9350 |
| 3 | Pahadi | N/A |
| 4 | Desh | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Barwa

**Top-1 Similarity:** 0.9362

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Barwa was the highest-ranked candidate in the TRF reference space for this recording.

## 010 — Sonogo Aian Dada
**Original Song ID:** BAUL_010

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9457 |
| 2 | Bahar | 0.9417 |
| 3 | Malkauns | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9457

**Top-1 vs Top-2 Margin:** 0.0040

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 011 — Chor Poreche Babur Bagane
**Original Song ID:** BAUL_011

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9281 |
| 2 | Megh | 0.9205 |
| 3 | Gauri | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | Abhogi | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9281

**Top-1 vs Top-2 Margin:** 0.0076

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 012 — Golemale Gole Male
**Original Song ID:** BAUL_012

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9357 |
| 2 | Jhinjhoti | 0.9341 |
| 3 | Abhogi | N/A |
| 4 | Bilaval | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9357

**Top-1 vs Top-2 Margin:** 0.0016

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 013 — Menokaa Maathaay Dilo Ghomtaa
**Original Song ID:** BAUL_013

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Puriya | 0.9328 |
| 2 | Jaunpuri | 0.9276 |
| 3 | Pahadi | N/A |
| 4 | Sohni | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Puriya

**Top-1 Similarity:** 0.9328

**Top-1 vs Top-2 Margin:** 0.0052

**Interpretation:**
Raag Puriya was the highest-ranked candidate in the TRF reference space for this recording.

## 014 — Dekhechhi Rup-Sagare
**Original Song ID:** BAUL_014

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9367 |
| 2 | Sorath | 0.9317 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9367

**Top-1 vs Top-2 Margin:** 0.0049

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 015 — Aami Kothay Pabo Tare
**Original Song ID:** BAUL_015

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Abhogi | 0.9181 |
| 2 | Bhairavi | 0.9172 |
| 3 | Malkauns | N/A |
| 4 | Asavari | N/A |
| 5 | Kedar | N/A |

**Top-1 Candidate:** Abhogi

**Top-1 Similarity:** 0.9181

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Abhogi was the highest-ranked candidate in the TRF reference space for this recording.

## 016 — Shesher Deene Sejon Bine
**Original Song ID:** BAUL_016

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9355 |
| 2 | Madhuvanti | 0.9304 |
| 3 | Darbari | N/A |
| 4 | Malkauns | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9355

**Top-1 vs Top-2 Margin:** 0.0051

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 017 — Milon Hobe Koto Dine
**Original Song ID:** BAUL_017

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhimpalasi | 0.9335 |
| 2 | Bhoopali | 0.9318 |
| 3 | Barwa | N/A |
| 4 | Desh | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Bhimpalasi

**Top-1 Similarity:** 0.9335

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Bhimpalasi was the highest-ranked candidate in the TRF reference space for this recording.

## 018 — Khanchar Bhitor Achin Pakhi
**Original Song ID:** BAUL_018

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Abhogi | 0.9294 |
| 2 | Yaman or Kalyan | 0.9289 |
| 3 | Kafi | N/A |
| 4 | Jhinjhoti | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Abhogi

**Top-1 Similarity:** 0.9294

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Abhogi was the highest-ranked candidate in the TRF reference space for this recording.

## 019 — Aami Ek Pagal Baul
**Original Song ID:** BAUL_019

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9208 |
| 2 | Ahir Bhairav | 0.9207 |
| 3 | Sorath | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9208

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 020 — Jemon Beni Temni Rabey
**Original Song ID:** BAUL_020

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9400 |
| 2 | Ramkali | 0.9357 |
| 3 | Khamaj | N/A |
| 4 | Desh | N/A |
| 5 | Megh Malhar | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9400

**Top-1 vs Top-2 Margin:** 0.0043

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 021 — Prem Koraa Ki Jwalaa
**Original Song ID:** BAUL_021

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9417 |
| 2 | Gauri | 0.9334 |
| 3 | Bageshree | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9417

**Top-1 vs Top-2 Margin:** 0.0084

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 022 — Emon Mon Byabsaa
**Original Song ID:** BAUL_022

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Durga | 0.9488 |
| 2 | Gauri | 0.9449 |
| 3 | Rageshree | N/A |
| 4 | Asavari | N/A |
| 5 | Bihag | N/A |

**Top-1 Candidate:** Durga

**Top-1 Similarity:** 0.9488

**Top-1 vs Top-2 Margin:** 0.0040

**Interpretation:**
Raag Durga was the highest-ranked candidate in the TRF reference space for this recording.

## 023 — Manush Bhajo Manusheyi Pujo
**Original Song ID:** BAUL_023

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9455 |
| 2 | N/A | 0.0000 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9455

**Top-1 vs Top-2 Margin:** 0.9455

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 024 — Dil Dariyar Majhete
**Original Song ID:** BAUL_024

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9334 |
| 2 | Bageshree | 0.9246 |
| 3 | Desh | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9334

**Top-1 vs Top-2 Margin:** 0.0088

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 025 — Bahu Punya Phale Krishna Mele
**Original Song ID:** BAUL_025

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9353 |
| 2 | Darbari | 0.9294 |
| 3 | Rageshree | N/A |
| 4 | Kedar | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9353

**Top-1 vs Top-2 Margin:** 0.0059

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 026 — Sokhi Jamuna Jol
**Original Song ID:** BAUL_026

**Artist:** Purna Das Baul

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9320 |
| 2 | Megh | 0.9315 |
| 3 | Yaman or Kalyan | N/A |
| 4 | Yaman Kalyan | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9320

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 027 — Pre Loye Jao Amay
**Original Song ID:** BAUL_027

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9434 |
| 2 | Bhairav | 0.9402 |
| 3 | Poorvi | N/A |
| 4 | Abhogi | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9434

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 028 — Somoy Gele Sadhon Hobena
**Original Song ID:** BAUL_028

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Nat Bhairav | 0.9180 |
| 2 | Megh | 0.9171 |
| 3 | Bageshree | N/A |
| 4 | Khamaj | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Nat Bhairav

**Top-1 Similarity:** 0.9180

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Nat Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 029 — Aar Amare Marishne Maa
**Original Song ID:** BAUL_029

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Miyan Ki Todi orTodi or Darbari Todi | 0.9414 |
| 2 | Bahar | 0.9394 |
| 3 | Desh | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Miyan Ki Todi orTodi or Darbari Todi

**Top-1 Similarity:** 0.9414

**Top-1 vs Top-2 Margin:** 0.0020

**Interpretation:**
Raag Miyan Ki Todi orTodi or Darbari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 030 — Dhonno Dhonno Boli Tare
**Original Song ID:** BAUL_030

**Artist:** Farida Parveen

**Genre:** Baul

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9370 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9339 |
| 3 | Bihag | N/A |
| 4 | Bageshree | N/A |
| 5 | Yaman Kalyan | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9370

**Top-1 vs Top-2 Margin:** 0.0031

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 031 — Tera Yaar Bolda
**Original Song ID:** BHANGRA_001

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9212 |
| 2 | Darbari | 0.9210 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Bibhas | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9212

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 032 — Jatt Di Pasand
**Original Song ID:** BHANGRA_002

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9307 |
| 2 | Bahar | 0.9272 |
| 3 | Asavari | N/A |
| 4 | Darbari | N/A |
| 5 | Yaman or Kalyan | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9307

**Top-1 vs Top-2 Margin:** 0.0035

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 033 — Dupatta Tera Satrang Da
**Original Song ID:** BHANGRA_003

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9340 |
| 2 | Alhaiya Bilaval | 0.9340 |
| 3 | Bibhas | N/A |
| 4 | Bahar | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9340

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 034 — Wang Wargi Kudi
**Original Song ID:** BHANGRA_004

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9286 |
| 2 | Abhogi | 0.9243 |
| 3 | Ahir Bhairav | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9286

**Top-1 vs Top-2 Margin:** 0.0043

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 035 — Sanu Tedi Tedi Takdi Tu
**Original Song ID:** BHANGRA_005

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman or Kalyan | 0.9279 |
| 2 | Megh | 0.9239 |
| 3 | Bahar | N/A |
| 4 | Khamaj | N/A |
| 5 | Marva | N/A |

**Top-1 Candidate:** Yaman or Kalyan

**Top-1 Similarity:** 0.9279

**Top-1 vs Top-2 Margin:** 0.0041

**Interpretation:**
Raag Yaman or Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 036 — Munda Ki Mangda
**Original Song ID:** BHANGRA_006

**Artist:** Surjit Bindrakhia

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9311 |
| 2 | Alhaiya Bilaval | 0.9282 |
| 3 | Bhairav Bahar | N/A |
| 4 | Jaunpuri | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9311

**Top-1 vs Top-2 Margin:** 0.0029

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 037 — Tunak Tunak Tun
**Original Song ID:** BHANGRA_007

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9487 |
| 2 | Deepak | 0.9359 |
| 3 | Bhatiyar | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | sarang | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9487

**Top-1 vs Top-2 Margin:** 0.0127

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 038 — Bolo Ta Ra Ra
**Original Song ID:** BHANGRA_008

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9417 |
| 2 | Pahadi | 0.9385 |
| 3 | Asavari | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9417

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 039 — Ho Jayegi Balle Balle
**Original Song ID:** BHANGRA_009

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9468 |
| 2 | Basant | 0.9397 |
| 3 | Asavari | N/A |
| 4 | Kedar | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9468

**Top-1 vs Top-2 Margin:** 0.0071

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 040 — Dardi Rab Rab Kardi
**Original Song ID:** BHANGRA_010

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9320 |
| 2 | Asavari | 0.9275 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9320

**Top-1 vs Top-2 Margin:** 0.0045

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 041 — Saade Dil Te Chhuriyan Chaliyan
**Original Song ID:** BHANGRA_011

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9344 |
| 2 | Ahir Bhairav | 0.9298 |
| 3 | Barwa | N/A |
| 4 | Deepak | N/A |
| 5 | Bibhas | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9344

**Top-1 vs Top-2 Margin:** 0.0046

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 042 — Na Na Na Na Na Re
**Original Song ID:** BHANGRA_012

**Artist:** Daler Mehndi

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9397 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 |
| 3 | Madhuvanti | N/A |
| 4 | Rageshree | N/A |
| 5 | Alhaiya Bilaval | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9397

**Top-1 vs Top-2 Margin:** 0.0015

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 043 — Pao Bhangra
**Original Song ID:** BHANGRA_013

**Artist:** Balwinder Safri

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9151 |
| 2 | Bahar | 0.9140 |
| 3 | Malhar | N/A |
| 4 | Megh Malhar | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9151

**Top-1 vs Top-2 Margin:** 0.0011

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 044 — Ambran Tu Ayi Hui Hoor Sohniye
**Original Song ID:** BHANGRA_014

**Artist:** Balwinder Safri

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Gujari Todi | 0.9384 |
| 2 | Megh | 0.9304 |
| 3 | Puriya | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Gujari Todi

**Top-1 Similarity:** 0.9384

**Top-1 vs Top-2 Margin:** 0.0081

**Interpretation:**
Raag Gujari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 045 — Sariyan Toon Sohni
**Original Song ID:** BHANGRA_015

**Artist:** Balwinder Safri

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9291 |
| 2 | Darbari | 0.9286 |
| 3 | Asavari | N/A |
| 4 | Puriya | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9291

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 046 — Mitran Ne Mundri Te
**Original Song ID:** BHANGRA_016

**Artist:** Balwinder Safri

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9400 |
| 2 | Puriya | 0.9394 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Asavari | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9400

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 047 — Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla)
**Original Song ID:** BHANGRA_017

**Artist:** Balwinder Safri

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Gujari Todi | 0.9387 |
| 2 | Bilaval | 0.9333 |
| 3 | Puriya | N/A |
| 4 | Megh | N/A |
| 5 | Alhaiya Bilaval | N/A |

**Top-1 Candidate:** Gujari Todi

**Top-1 Similarity:** 0.9387

**Top-1 vs Top-2 Margin:** 0.0054

**Interpretation:**
Raag Gujari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 048 — Tutak Tutak Tutiya
**Original Song ID:** BHANGRA_018

**Artist:** Malkit Singh (Golden Star)

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9297 |
| 2 | Alhaiya Bilaval | 0.9295 |
| 3 | Desh | N/A |
| 4 | Asavari | N/A |
| 5 | Gujari Todi | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9297

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 049 — Kurri Garam Jayee
**Original Song ID:** BHANGRA_019

**Artist:** Malkit Singh (Golden Star)

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Rageshree | 0.9322 |
| 2 | Puriya | 0.9319 |
| 3 | Gauri | N/A |
| 4 | Kedar | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Rageshree

**Top-1 Similarity:** 0.9322

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Rageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 050 — Chal Hun
**Original Song ID:** BHANGRA_020

**Artist:** Malkit Singh (Golden Star)

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Shree | 0.9341 |
| 2 | Gujari Todi | 0.9327 |
| 3 | Bhimpalasi | N/A |
| 4 | Darbari | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Shree

**Top-1 Similarity:** 0.9341

**Top-1 vs Top-2 Margin:** 0.0014

**Interpretation:**
Raag Shree was the highest-ranked candidate in the TRF reference space for this recording.

## 051 — Kinni Sohni
**Original Song ID:** BHANGRA_021

**Artist:** Malkit Singh (Golden Star)

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9421 |
| 2 | Alhaiya Bilaval | 0.9358 |
| 3 | Rageshree | N/A |
| 4 | Adana | N/A |
| 5 | Desi | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9421

**Top-1 vs Top-2 Margin:** 0.0063

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 052 — Mera Yaar Vajaye Dhol
**Original Song ID:** BHANGRA_022

**Artist:** Apna Sangeet

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9433 |
| 2 | Malkauns | 0.9424 |
| 3 | Yaman Kalyan | N/A |
| 4 | Sorath | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9433

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 053 — Bhaina Nachiyaan
**Original Song ID:** BHANGRA_023

**Artist:** Apna Sangeet

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9301 |
| 2 | Puriya | 0.9252 |
| 3 | Asavari | N/A |
| 4 | Barwa | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9301

**Top-1 vs Top-2 Margin:** 0.0049

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 054 — Apna Sangeet Vajeh Apna Sangeet
**Original Song ID:** BHANGRA_024

**Artist:** Apna Sangeet

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9226 |
| 2 | Desh | 0.9145 |
| 3 | Ahir Bhairav | N/A |
| 4 | Bageshree | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9226

**Top-1 vs Top-2 Margin:** 0.0081

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 055 — Soho Road Utey
**Original Song ID:** BHANGRA_025

**Artist:** Apna Sangeet

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9347 |
| 2 | Bilaval | 0.9321 |
| 3 | Desh | N/A |
| 4 | Darbari | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9347

**Top-1 vs Top-2 Margin:** 0.0026

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 056 — Kuriyan Ne Paya Bhangra
**Original Song ID:** BHANGRA_026

**Artist:** Kuldeep Manak

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9309 |
| 2 | Asavari | 0.9244 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Madhuvanti | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9309

**Top-1 vs Top-2 Margin:** 0.0065

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 057 — Jodi (Big Day Party) ft. Panjabi MC
**Original Song ID:** BHANGRA_027

**Artist:** Kuldeep Manak

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9342 |
| 2 | Asavari | 0.9312 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9342

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 058 — Jugni
**Original Song ID:** BHANGRA_028

**Artist:** Kuldeep Manak

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9351 |
| 2 | Sorath | 0.9308 |
| 3 | Adana | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9351

**Top-1 vs Top-2 Margin:** 0.0043

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 059 — Putt Jattan De
**Original Song ID:** BHANGRA_029

**Artist:** Surinder Shinda

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Abhogi | 0.9170 |
| 2 | Darbari | 0.9168 |
| 3 | Pahadi | N/A |
| 4 | Bhairavi | N/A |
| 5 | Desi | N/A |

**Top-1 Candidate:** Abhogi

**Top-1 Similarity:** 0.9170

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Abhogi was the highest-ranked candidate in the TRF reference space for this recording.

## 060 — Sucha Soorma (Balbiro Bhabhi)
**Original Song ID:** BHANGRA_030

**Artist:** Surinder Shinda

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | suddha kalyan | 0.9264 |
| 2 | Purba | 0.9229 |
| 3 | Adana | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** suddha kalyan

**Top-1 Similarity:** 0.9264

**Top-1 vs Top-2 Margin:** 0.0035

**Interpretation:**
Raag suddha kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 061 — Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya)
**Original Song ID:** BHANGRA_031

**Artist:** Surinder Shinda

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Rageshree | 0.9379 |
| 2 | Khamaj | 0.9289 |
| 3 | Bilaval | N/A |
| 4 | Sorath | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Rageshree

**Top-1 Similarity:** 0.9379

**Top-1 vs Top-2 Margin:** 0.0090

**Interpretation:**
Raag Rageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 062 — Jeona Morh
**Original Song ID:** BHANGRA_032

**Artist:** Surinder Shinda

**Genre:** Bhangra

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant Mukhari | 0.9448 |
| 2 | Bhairav | 0.9441 |
| 3 | Basant | N/A |
| 4 | Megh Malhar | N/A |
| 5 | Malhar | N/A |

**Top-1 Candidate:** Basant Mukhari

**Top-1 Similarity:** 0.9448

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Basant Mukhari was the highest-ranked candidate in the TRF reference space for this recording.

## 063 — Morom Logai Logai
**Original Song ID:** BIHU_001

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sohni | 0.9355 |
| 2 | Bahar | 0.9334 |
| 3 | Megh Malhar | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Sohni

**Top-1 Similarity:** 0.9355

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Sohni was the highest-ranked candidate in the TRF reference space for this recording.

## 064 — Toi Nagini Ne
**Original Song ID:** BIHU_002

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9280 |
| 2 | Puriya | 0.9271 |
| 3 | Sohni | N/A |
| 4 | Bahar | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9280

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 065 — Kokalote Tongali
**Original Song ID:** BIHU_003

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Hindol | 0.9402 |
| 2 | Sorath | 0.9400 |
| 3 | Madhuvanti | N/A |
| 4 | suddha kalyan | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Hindol

**Top-1 Similarity:** 0.9402

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Hindol was the highest-ranked candidate in the TRF reference space for this recording.

## 066 — Tumake Logot
**Original Song ID:** BIHU_004

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9349 |
| 2 | Puriya | 0.9349 |
| 3 | Malkauns | N/A |
| 4 | Darbari | N/A |
| 5 | Abhogi | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9349

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 067 — Kinu Sawonire Sala
**Original Song ID:** BIHU_005

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desi | 0.9378 |
| 2 | Darbari | 0.9354 |
| 3 | Hindol | N/A |
| 4 | Desh | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Desi

**Top-1 Similarity:** 0.9378

**Top-1 vs Top-2 Margin:** 0.0024

**Interpretation:**
Raag Desi was the highest-ranked candidate in the TRF reference space for this recording.

## 068 — Dikhow Noi Eribo
**Original Song ID:** BIHU_006

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9435 |
| 2 | Sorath | 0.9399 |
| 3 | Marva | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9435

**Top-1 vs Top-2 Margin:** 0.0036

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 069 — Phool Phulise
**Original Song ID:** BIHU_007

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9279 |
| 2 | Pahadi | 0.9243 |
| 3 | Malhar | N/A |
| 4 | Darbari | N/A |
| 5 | Desi | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9279

**Top-1 vs Top-2 Margin:** 0.0037

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 070 — Porbotor Dhekiya
**Original Song ID:** BIHU_008

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9392 |
| 2 | Desi | 0.9379 |
| 3 | Nat Bhairav | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Barwa | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9392

**Top-1 vs Top-2 Margin:** 0.0013

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 071 — O Nahor
**Original Song ID:** BIHU_009

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9373 |
| 2 | Bahar | 0.9318 |
| 3 | Puriya | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9373

**Top-1 vs Top-2 Margin:** 0.0055

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 072 — Janmoni Xopunote Ahiba
**Original Song ID:** BIHU_010

**Artist:** Zubeen Garg

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaskhani Todi | 0.9465 |
| 2 | Purba | 0.9454 |
| 3 | Hindol | N/A |
| 4 | Asavari | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Bilaskhani Todi

**Top-1 Similarity:** 0.9465

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Bilaskhani Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 073 — Eijon Korenu Nasoni
**Original Song ID:** BIHU_011

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9127 |
| 2 | Bhimpalasi | 0.9113 |
| 3 | Rageshree | N/A |
| 4 | suddha kalyan | N/A |
| 5 | Komal Bhairav | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9127

**Top-1 vs Top-2 Margin:** 0.0014

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 074 — Laje Moru Moru
**Original Song ID:** BIHU_012

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9350 |
| 2 | Puriya | 0.9348 |
| 3 | Darbari | N/A |
| 4 | Asavari | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9350

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 075 — Ki Sai Aso
**Original Song ID:** BIHU_013

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9378 |
| 2 | Asavari | 0.9377 |
| 3 | Adana | N/A |
| 4 | Bhoopali | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9378

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 076 — Bihureno Birina
**Original Song ID:** BIHU_014

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9399 |
| 2 | Bahar | 0.9386 |
| 3 | Khamaj | N/A |
| 4 | Rageshree | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9399

**Top-1 vs Top-2 Margin:** 0.0013

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 077 — Duetar Boraghor
**Original Song ID:** BIHU_015

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9327 |
| 2 | Asavari | 0.9314 |
| 3 | Jhinjhoti | N/A |
| 4 | Darbari | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9327

**Top-1 vs Top-2 Margin:** 0.0013

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 078 — Sangor Chira Pitha
**Original Song ID:** BIHU_016

**Artist:** Bhupen Hazarika

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9341 |
| 2 | Yaman or Kalyan | 0.9337 |
| 3 | Bhoopali | N/A |
| 4 | Puriya | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9341

**Top-1 vs Top-2 Margin:** 0.0004

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 079 — Kopou Phul Phulile
**Original Song ID:** BIHU_017

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9308 |
| 2 | Puriya | 0.9258 |
| 3 | Barwa | N/A |
| 4 | Megh | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9308

**Top-1 vs Top-2 Margin:** 0.0050

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 080 — Amona Dhanoke
**Original Song ID:** BIHU_018

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9321 |
| 2 | Megh | 0.9310 |
| 3 | Khamaj | N/A |
| 4 | Asavari | N/A |
| 5 | Bhupal Todi | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9321

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 081 — O Dhei Oi
**Original Song ID:** BIHU_019

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9416 |
| 2 | Bhairavi | 0.9335 |
| 3 | Desh | N/A |
| 4 | Bahar | N/A |
| 5 | Bageshree | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9416

**Top-1 vs Top-2 Margin:** 0.0081

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 082 — Dhansiri Pare Hoi
**Original Song ID:** BIHU_020

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9316 |
| 2 | Ahir Bhairav | 0.9304 |
| 3 | Shree | N/A |
| 4 | Bhatiyar | N/A |
| 5 | Lalit | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9316

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 083 — Petore Juruni Bhat
**Original Song ID:** BIHU_021

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh Malhar | 0.9354 |
| 2 | Malhar | 0.9354 |
| 3 | Adana | N/A |
| 4 | Bilaval | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Megh Malhar

**Top-1 Similarity:** 0.9354

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Megh Malhar was the highest-ranked candidate in the TRF reference space for this recording.

## 084 — Aag Fale Dekhilo
**Original Song ID:** BIHU_022

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9400 |
| 2 | Deepak | 0.9375 |
| 3 | Bhairavi | N/A |
| 4 | Rageshree | N/A |
| 5 | Bilaval | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9400

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 085 — Aghunor Potharot
**Original Song ID:** BIHU_023

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9258 |
| 2 | Sorath | 0.9236 |
| 3 | sarang | N/A |
| 4 | Darbari | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9258

**Top-1 vs Top-2 Margin:** 0.0022

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 086 — Ebela Hale Bau
**Original Song ID:** BIHU_024

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9140 |
| 2 | Abhogi | 0.9121 |
| 3 | Megh | N/A |
| 4 | Yaman or Kalyan | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9140

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 087 — Senai o Tumake Bisari
**Original Song ID:** BIHU_025

**Artist:** Khagen Mahanta

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9319 |
| 2 | Megh | 0.9283 |
| 3 | Bahar | N/A |
| 4 | Bilaval | N/A |
| 5 | Rageshree | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9319

**Top-1 vs Top-2 Margin:** 0.0036

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 088 — Bohag Ahil Buli
**Original Song ID:** BIHU_026

**Artist:** Shanta Uzir

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9448 |
| 2 | Bhoopali | 0.9410 |
| 3 | Sohni | N/A |
| 4 | Adana | N/A |
| 5 | Bhimpalasi | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9448

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 089 — Tumi Nasoni
**Original Song ID:** BIHU_027

**Artist:** Shanta Uzir

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9331 |
| 2 | Yaman Kalyan | 0.9299 |
| 3 | Darbari | N/A |
| 4 | Sohni | N/A |
| 5 | Abhogi | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9331

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 090 — Xunjone Nuxudhiba
**Original Song ID:** BIHU_028

**Artist:** Zubeen Garg, Vitali Das

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9347 |
| 2 | Bahar | 0.9335 |
| 3 | Ahir Bhairav | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9347

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 091 — Morone Phulonit
**Original Song ID:** BIHU_029

**Artist:** Zubeen Garg, Vitali Das

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhatiyar | 0.9391 |
| 2 | Darbari | 0.9380 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bhatiyar

**Top-1 Similarity:** 0.9391

**Top-1 vs Top-2 Margin:** 0.0011

**Interpretation:**
Raag Bhatiyar was the highest-ranked candidate in the TRF reference space for this recording.

## 092 — Nila Nila Akhore
**Original Song ID:** BIHU_030

**Artist:** Zubeen Garg, Nirmali Das

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaskhani Todi | 0.9336 |
| 2 | Bahar | 0.9333 |
| 3 | Adana | N/A |
| 4 | Puriya | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Bilaskhani Todi

**Top-1 Similarity:** 0.9336

**Top-1 vs Top-2 Margin:** 0.0004

**Interpretation:**
Raag Bilaskhani Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 093 — Kahili puwate
**Original Song ID:** BIHU_031

**Artist:** Zubeen Garg, Priyanka Bharali

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9409 |
| 2 | Darbari | 0.9390 |
| 3 | suddha kalyan | N/A |
| 4 | Hindol | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9409

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 094 — Rohare O Rohedoi
**Original Song ID:** BIHU_032

**Artist:** Bhitali Das

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sohni | 0.9367 |
| 2 | Khamaj | 0.9349 |
| 3 | Bilaval | N/A |
| 4 | Abhogi | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Sohni

**Top-1 Similarity:** 0.9367

**Top-1 vs Top-2 Margin:** 0.0018

**Interpretation:**
Raag Sohni was the highest-ranked candidate in the TRF reference space for this recording.

## 095 — Dhengkoli
**Original Song ID:** BIHU_033

**Artist:** Navanita Sharma

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9378 |
| 2 | Asavari | 0.9355 |
| 3 | Madhuvanti | N/A |
| 4 | Yaman Kalyan | N/A |
| 5 | Bhimpalasi | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9378

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 096 — Gabhoru Bihu
**Original Song ID:** BIHU_034

**Artist:** Deeplina Deka

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9449 |
| 2 | Bhairav | 0.9412 |
| 3 | Bihag | N/A |
| 4 | Darbari | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9449

**Top-1 vs Top-2 Margin:** 0.0037

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 097 — Bohagi
**Original Song ID:** BIHU_035

**Artist:** Rupankrita and Alankrita

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9335 |
| 2 | Adana | 0.9314 |
| 3 | Megh | N/A |
| 4 | Khamaj | N/A |
| 5 | Kedar | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9335

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 098 — Podulit Bihu
**Original Song ID:** BIHU_036

**Artist:** Rupankrita and Alankrita

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9383 |
| 2 | Desh | 0.9361 |
| 3 | Khamaj | N/A |
| 4 | Darbari | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9383

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 099 — Tok Dekhi Mor Gaa
**Original Song ID:** BIHU_037

**Artist:** Abhishruti Bezbaruah

**Genre:** Bihu

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Madhuvanti | 0.9381 |
| 2 | Alhaiya Bilaval | 0.9371 |
| 3 | Megh | N/A |
| 4 | Puriya | N/A |
| 5 | Jaunpuri | N/A |

**Top-1 Candidate:** Madhuvanti

**Top-1 Similarity:** 0.9381

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Madhuvanti was the highest-ranked candidate in the TRF reference space for this recording.

## 100 — জাগাইওনা মোরে সই গো
**Original Song ID:** DHAMAIL_001

**Artist:** Dithi Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9274 |
| 2 | Madhuvanti | 0.9255 |
| 3 | Poorvi | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9274

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 101 — আর যাবো না জলে
**Original Song ID:** DHAMAIL_002

**Artist:** Dithi Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9325 |
| 2 | Madhuvanti | 0.9313 |
| 3 | Poorvi | N/A |
| 4 | Darbari | N/A |
| 5 | Jaunpuri | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9325

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 102 — বন্ধু অতও যাইবায় গিয়া
**Original Song ID:** DHAMAIL_003

**Artist:** Dithi Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9343 |
| 2 | Khamaj | 0.9335 |
| 3 | Kedar | N/A |
| 4 | Lalit | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9343

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 103 — ধামাইল দেও গো চড়চড়ির মা
**Original Song ID:** DHAMAIL_004

**Artist:** Dithi Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Lalit | 0.9354 |
| 2 | Bhatiyar | 0.9345 |
| 3 | Bhairav | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Lalit

**Top-1 Similarity:** 0.9354

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Lalit was the highest-ranked candidate in the TRF reference space for this recording.

## 104 — বাঁশি বাজগো রইয়া রইয়া
**Original Song ID:** DHAMAIL_005

**Artist:** Dithi Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9257 |
| 2 | Nat Bhairav | 0.9242 |
| 3 | Darbari | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | Bageshree | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9257

**Top-1 vs Top-2 Margin:** 0.0014

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 105 — যাইও না রে প্রানো বন্ধু আমারে বুলিয়া
**Original Song ID:** DHAMAIL_006

**Artist:** Uma Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9376 |
| 2 | Malkauns | 0.9354 |
| 3 | Megh | N/A |
| 4 | sarang | N/A |
| 5 | Yaman Kalyan | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9376

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 106 — দেইখে আইলাম তারে সইগো
**Original Song ID:** DHAMAIL_007

**Artist:** Uma Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Abhogi | 0.9279 |
| 2 | sarang | 0.9258 |
| 3 | Jaunpuri | N/A |
| 4 | Desh | N/A |
| 5 | Komal Bhairav | N/A |

**Top-1 Candidate:** Abhogi

**Top-1 Similarity:** 0.9279

**Top-1 vs Top-2 Margin:** 0.0020

**Interpretation:**
Raag Abhogi was the highest-ranked candidate in the TRF reference space for this recording.

## 107 — মনের জ্বালায় জ্বলে মরি
**Original Song ID:** DHAMAIL_008

**Artist:** Uma Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9486 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9476 |
| 3 | Purba | N/A |
| 4 | Bilaval | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9486

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 108 — ধামাইল শেষ হইয়া যায়
**Original Song ID:** DHAMAIL_009

**Artist:** Uma Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant Mukhari | 0.9399 |
| 2 | Malkauns | 0.9392 |
| 3 | sarang | N/A |
| 4 | Bhairav Bahar | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Basant Mukhari

**Top-1 Similarity:** 0.9399

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Basant Mukhari was the highest-ranked candidate in the TRF reference space for this recording.

## 109 — আমরা কার বাড়িত আইলাম গো
**Original Song ID:** DHAMAIL_010

**Artist:** Uma Das

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Jaunpuri | 0.9390 |
| 2 | Purba | 0.9356 |
| 3 | Gujari Todi | N/A |
| 4 | Megh | N/A |
| 5 | sarang | N/A |

**Top-1 Candidate:** Jaunpuri

**Top-1 Similarity:** 0.9390

**Top-1 vs Top-2 Margin:** 0.0035

**Interpretation:**
Raag Jaunpuri was the highest-ranked candidate in the TRF reference space for this recording.

## 110 — বল গো বল সখি
**Original Song ID:** DHAMAIL_011

**Artist:** Habib Simu/Shimu

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9381 |
| 2 | Sorath | 0.9374 |
| 3 | Bhimpalasi | N/A |
| 4 | Darbari | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9381

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 111 — ইতা কিতা করে গো মনে
**Original Song ID:** DHAMAIL_012

**Artist:** Habib Simu/Shimu

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9444 |
| 2 | Multani | 0.9407 |
| 3 | Purba | N/A |
| 4 | Sorath | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9444

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 112 — শুনোগো শুনোগো সখি
**Original Song ID:** DHAMAIL_013

**Artist:** Habib Simu/Shimu

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9474 |
| 2 | Bageshree | 0.9434 |
| 3 | Asavari | N/A |
| 4 | Khamaj | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9474

**Top-1 vs Top-2 Margin:** 0.0040

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 113 — শ্যামরে এখন আর কাঁদিও না
**Original Song ID:** DHAMAIL_014

**Artist:** Habib Simu/Shimu

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9258 |
| 2 | Alhaiya Bilaval | 0.9233 |
| 3 | Madhuvanti | N/A |
| 4 | Adana | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9258

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 114 — চান্নি রাইতে আইও কালা
**Original Song ID:** DHAMAIL_015

**Artist:** Habib Simu/Shimu

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Madhuvanti | 0.9452 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 |
| 3 | Multani | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Madhuvanti

**Top-1 Similarity:** 0.9452

**Top-1 vs Top-2 Margin:** 0.0070

**Interpretation:**
Raag Madhuvanti was the highest-ranked candidate in the TRF reference space for this recording.

## 115 — যুগল মিলন হইলো গো
**Original Song ID:** DHAMAIL_016

**Artist:** Jumiar Shanta

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bihag | 0.9409 |
| 2 | Kedar | 0.9408 |
| 3 | Multani | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Bihag

**Top-1 Similarity:** 0.9409

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Bihag was the highest-ranked candidate in the TRF reference space for this recording.

## 116 — বলো গো বলো সই
**Original Song ID:** DHAMAIL_017

**Artist:** Pausali Banerjee

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9351 |
| 2 | Bhoopali | 0.9330 |
| 3 | Yaman Kalyan | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9351

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 117 — কত আদরে আদরে
**Original Song ID:** DHAMAIL_018

**Artist:** Pausali Banerjee

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhupal Todi | 0.9340 |
| 2 | Desh | 0.9286 |
| 3 | Yaman Kalyan | N/A |
| 4 | Barwa | N/A |
| 5 | Poorvi | N/A |

**Top-1 Candidate:** Bhupal Todi

**Top-1 Similarity:** 0.9340

**Top-1 vs Top-2 Margin:** 0.0053

**Interpretation:**
Raag Bhupal Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 118 — জলে গিয়েছিলাম সই
**Original Song ID:** DHAMAIL_019

**Artist:** Pausali Banerjee

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9376 |
| 2 | Asavari | 0.9334 |
| 3 | Adana | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9376

**Top-1 vs Top-2 Margin:** 0.0042

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 119 — বন্ধু বিনে প্রাণ বাঁচে না
**Original Song ID:** DHAMAIL_020

**Artist:** Pausali Banerjee

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9275 |
| 2 | Asavari | 0.9245 |
| 3 | Sorath | N/A |
| 4 | Khamaj | N/A |
| 5 | Bageshree | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9275

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 120 — জলের ঘাটে
**Original Song ID:** DHAMAIL_021

**Artist:** Pausali Banerjee

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bihag | 0.9414 |
| 2 | Yaman Kalyan | 0.9299 |
| 3 | Kedar | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bihag

**Top-1 Similarity:** 0.9414

**Top-1 vs Top-2 Margin:** 0.0115

**Interpretation:**
Raag Bihag was the highest-ranked candidate in the TRF reference space for this recording.

## 121 — Bhalo Kare Bajam Re
**Original Song ID:** DHAMAIL_022

**Artist:** Maadal

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Poorvi | 0.9384 |
| 2 | Darbari | 0.9369 |
| 3 | Lalit | N/A |
| 4 | Desh | N/A |
| 5 | Bahar | N/A |

**Top-1 Candidate:** Poorvi

**Top-1 Similarity:** 0.9384

**Top-1 vs Top-2 Margin:** 0.0015

**Interpretation:**
Raag Poorvi was the highest-ranked candidate in the TRF reference space for this recording.

## 122 — Tara Lowo Go Jol
**Original Song ID:** DHAMAIL_023

**Artist:** Maadal

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9349 |
| 2 | Bilaskhani Todi | 0.9298 |
| 3 | Bilaval | N/A |
| 4 | Kedar | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9349

**Top-1 vs Top-2 Margin:** 0.0052

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 123 — Jhiko Jhiko Khodi Re
**Original Song ID:** DHAMAIL_024

**Artist:** Maadal

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desi | 0.9446 |
| 2 | Megh | 0.9420 |
| 3 | Pahadi | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Desi

**Top-1 Similarity:** 0.9446

**Top-1 vs Top-2 Margin:** 0.0026

**Interpretation:**
Raag Desi was the highest-ranked candidate in the TRF reference space for this recording.

## 124 — Nrittangon Sunamganj
**Original Song ID:** DHAMAIL_025

**Artist:** Debdas Chowdhury

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9243 |
| 2 | Malkauns | 0.9217 |
| 3 | Madhuvanti | N/A |
| 4 | Sorath | N/A |
| 5 | Rageshree | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9243

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 125 — জলার জলার ঘাটে
**Original Song ID:** DHAMAIL_026

**Artist:** Himel Dev Rahul

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9157 |
| 2 | Ahir Bhairav | 0.9156 |
| 3 | Gauri | N/A |
| 4 | Rageshree | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9157

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 126 — অচেনা এক রূপবতী
**Original Song ID:** DHAMAIL_027

**Artist:** Samim Siddiki

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9444 |
| 2 | Adana | 0.9435 |
| 3 | Megh | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9444

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 127 — কুঞ্জের মাঝে
**Original Song ID:** DHAMAIL_028

**Artist:** Madol Folk Band

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9370 |
| 2 | Sorath | 0.9343 |
| 3 | Yaman or Kalyan | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9370

**Top-1 vs Top-2 Margin:** 0.0027

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 128 — আজ রাধার কুঞ্জে
**Original Song ID:** DHAMAIL_029

**Artist:** Suman Dewan

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9412 |
| 2 | Sorath | 0.9384 |
| 3 | Pahadi | N/A |
| 4 | Malkauns | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9412

**Top-1 vs Top-2 Margin:** 0.0028

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 129 — জানো নি গো সই
**Original Song ID:** DHAMAIL_030

**Artist:** Jabel Raza

**Genre:** Dhamail

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9140 |
| 2 | Puriya | 0.9119 |
| 3 | Adana | N/A |
| 4 | Sorath | N/A |
| 5 | Alhaiya Bilaval | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9140

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 130 — Maa Pava Te Gadh Thi Utarya
**Original Song ID:** GARBA_001

**Artist:** Diwaliben Bhil

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9390 |
| 2 | Darbari | 0.9384 |
| 3 | Sorath | N/A |
| 4 | Asavari | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9390

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 131 — Madi Tara Mandiriye Ghantarav Thay
**Original Song ID:** GARBA_002

**Artist:** Vibha Desai, Sharavan Madhurai

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9412 |
| 2 | Bilaskhani Todi | 0.9394 |
| 3 | Basant Mukhari | N/A |
| 4 | Bhupal Todi | N/A |
| 5 | Bhairav Bahar | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9412

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 132 — Ghor Andhari Re Rataldi Maa
**Original Song ID:** GARBA_003

**Artist:** Geeta Rabari

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9447 |
| 2 | Lalit | 0.9437 |
| 3 | Darbari | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9447

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 133 — Meldimaa Na Kumkum Pagla
**Original Song ID:** GARBA_004

**Artist:** Lalita Ghodadra

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9466 |
| 2 | suddha kalyan | 0.9457 |
| 3 | Desh | N/A |
| 4 | Sorath | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9466

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 134 — Amba Aavo To Ramiye
**Original Song ID:** GARBA_005

**Artist:** Lalita Ghodadra

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9489 |
| 2 | Yaman or Kalyan | 0.9442 |
| 3 | Megh | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9489

**Top-1 vs Top-2 Margin:** 0.0047

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 135 — Sonal Garbo Shire
**Original Song ID:** GARBA_006

**Artist:** Kishor Manraja

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh Malhar | 0.9469 |
| 2 | Komal Bhairav | 0.9466 |
| 3 | Gujari Todi | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Megh Malhar

**Top-1 Similarity:** 0.9469

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Megh Malhar was the highest-ranked candidate in the TRF reference space for this recording.

## 136 — Maa No Garbo Re
**Original Song ID:** GARBA_007

**Artist:** Geeta Rabari

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9410 |
| 2 | Darbari | 0.9401 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9410

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 137 — Unchi Talavdi Ni Kor
**Original Song ID:** GARBA_008

**Artist:** Asha Bhosle

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9397 |
| 2 | Marva | 0.9344 |
| 3 | Kafi | N/A |
| 4 | Bahar | N/A |
| 5 | Hemant | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9397

**Top-1 vs Top-2 Margin:** 0.0053

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 138 — Aavya Maa Na Norta
**Original Song ID:** GARBA_009

**Artist:** Pamela Jain, Divya Kumar

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9572 |
| 2 | Marva | 0.9544 |
| 3 | Rageshree | N/A |
| 4 | Bahar | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9572

**Top-1 vs Top-2 Margin:** 0.0027

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 139 — Khodiyar Maa Nu Holdu
**Original Song ID:** GARBA_010

**Artist:** Kinjal Dave

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9336 |
| 2 | Bhimpalasi | 0.9319 |
| 3 | Darbari | N/A |
| 4 | Marva | N/A |
| 5 | Lalit | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9336

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 140 — Sathiya Puravo
**Original Song ID:** GARBA_011

**Artist:** Damyanti Bardai

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9418 |
| 2 | Barwa | 0.9401 |
| 3 | Puriya | N/A |
| 4 | Yaman Kalyan | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9418

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 141 — Amba Aave Aarasur Vadi
**Original Song ID:** GARBA_012

**Artist:** Meena Patel

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman or Kalyan | 0.9249 |
| 2 | Khamaj | 0.9193 |
| 3 | Bihag | N/A |
| 4 | Puriya | N/A |
| 5 | Megh | N/A |

**Top-1 Candidate:** Yaman or Kalyan

**Top-1 Similarity:** 0.9249

**Top-1 vs Top-2 Margin:** 0.0055

**Interpretation:**
Raag Yaman or Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 142 — Tara Vina Maa Mane Sunu Lage
**Original Song ID:** GARBA_013

**Artist:** Meena Patel, Praful Dave

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9419 |
| 2 | Puriya | 0.9411 |
| 3 | Bilaskhani Todi | N/A |
| 4 | Bhairav Bahar | N/A |
| 5 | Gujari Todi | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9419

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 143 — Garba No Rang Lagiyo
**Original Song ID:** GARBA_014

**Artist:** Meena Patel, Praful Dave

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 |
| 2 | Durga | 0.9379 |
| 3 | Adana | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Jaunpuri | N/A |

**Top-1 Candidate:** Miyan Ki Todi orTodi or Darbari Todi

**Top-1 Similarity:** 0.9382

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Miyan Ki Todi orTodi or Darbari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 144 — Hindole Zule Amba
**Original Song ID:** GARBA_015

**Artist:** Meena Patel

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9291 |
| 2 | Barwa | 0.9284 |
| 3 | Darbari | N/A |
| 4 | Bilaval | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9291

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 145 — Jay Ambe Jagdambe Maa
**Original Song ID:** GARBA_016

**Artist:** Mital Gadhvi

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9402 |
| 2 | Bhoopali | 0.9379 |
| 3 | Megh | N/A |
| 4 | Bhairavi | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9402

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 146 — Halaji Tara Hath Vakhanu
**Original Song ID:** GARBA_017

**Artist:** Aditya Gadhvi

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9499 |
| 2 | Asavari | 0.9451 |
| 3 | Adana | N/A |
| 4 | Pahadi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9499

**Top-1 vs Top-2 Margin:** 0.0048

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 147 — Dholida Dhol Re Vagad
**Original Song ID:** GARBA_018

**Artist:** Farida Mir

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9344 |
| 2 | Bhimpalasi | 0.9337 |
| 3 | Asavari | N/A |
| 4 | Puriya | N/A |
| 5 | sarang | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9344

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 148 — Pankhida O Pankhida
**Original Song ID:** GARBA_019

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Jaunpuri | 0.9259 |
| 2 | Lalit | 0.9243 |
| 3 | Desh | N/A |
| 4 | Durga | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Jaunpuri

**Top-1 Similarity:** 0.9259

**Top-1 vs Top-2 Margin:** 0.0016

**Interpretation:**
Raag Jaunpuri was the highest-ranked candidate in the TRF reference space for this recording.

## 149 — Tu To Kali Ne Kalyani
**Original Song ID:** GARBA_020

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Barwa | 0.9305 |
| 2 | Bhairavi | 0.9281 |
| 3 | Yaman Kalyan | N/A |
| 4 | Bihag | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Barwa

**Top-1 Similarity:** 0.9305

**Top-1 vs Top-2 Margin:** 0.0024

**Interpretation:**
Raag Barwa was the highest-ranked candidate in the TRF reference space for this recording.

## 150 — He Jagjanani
**Original Song ID:** GARBA_021

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Kafi | 0.9288 |
| 2 | Desh | 0.9251 |
| 3 | Yaman Kalyan | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Kafi

**Top-1 Similarity:** 0.9288

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Kafi was the highest-ranked candidate in the TRF reference space for this recording.

## 151 — Chotile Dakla Vagya
**Original Song ID:** GARBA_022

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9191 |
| 2 | Basant Mukhari | 0.9151 |
| 3 | Megh Malhar | N/A |
| 4 | Malhar | N/A |
| 5 | Durga | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9191

**Top-1 vs Top-2 Margin:** 0.0040

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 152 — Jay Aadhyashakti
**Original Song ID:** GARBA_023

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9348 |
| 2 | N/A | 0.0000 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9348

**Top-1 vs Top-2 Margin:** 0.9348

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 153 — Dam Dam Dakla Vagya
**Original Song ID:** GARBA_024

**Artist:** Hemant Chauhan

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9234 |
| 2 | Gauri | 0.9228 |
| 3 | Basant Mukhari | N/A |
| 4 | Khamaj | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9234

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 154 — Sonala Vatakadi
**Original Song ID:** GARBA_025

**Artist:** Praful Dave, Damyanti Bardai

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9513 |
| 2 | Bilaval | 0.9485 |
| 3 | Kedar | N/A |
| 4 | Bhairav | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9513

**Top-1 vs Top-2 Margin:** 0.0028

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 155 — Kaho Poonam Na Chand
**Original Song ID:** GARBA_026

**Artist:** Shailendra Bharti

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9549 |
| 2 | Alhaiya Bilaval | 0.9547 |
| 3 | Asavari | N/A |
| 4 | Malkauns | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9549

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 156 — Aaj Ni Ghadi Te Raliyamani
**Original Song ID:** GARBA_027

**Artist:** Praful Dave, Meena Patel

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant | 0.9439 |
| 2 | Desh | 0.9421 |
| 3 | Jaunpuri | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Bhinna Shadja | N/A |

**Top-1 Candidate:** Basant

**Top-1 Similarity:** 0.9439

**Top-1 vs Top-2 Margin:** 0.0018

**Interpretation:**
Raag Basant was the highest-ranked candidate in the TRF reference space for this recording.

## 157 — Lejo Rasiya Re Rumal
**Original Song ID:** GARBA_028

**Artist:** Arvind Barot, Vatsala Patil

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9475 |
| 2 | Asavari | 0.9436 |
| 3 | Sorath | N/A |
| 4 | Jaunpuri | N/A |
| 5 | Bhimpalasi | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9475

**Top-1 vs Top-2 Margin:** 0.0039

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 158 — Pavana Dungre
**Original Song ID:** GARBA_029

**Artist:** Praful Dave

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9384 |
| 2 | Desi | 0.9379 |
| 3 | Bihag | N/A |
| 4 | Kafi | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9384

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 159 — Momai Na Pare Parevado Dhume Re
**Original Song ID:** GARBA_030

**Artist:** Arvind Barot

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9388 |
| 2 | Pahadi | 0.9380 |
| 3 | Gauri | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9388

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 160 — Rumzum Sandhani Aave Re Maa
**Original Song ID:** GARBA_031

**Artist:** Arvind Barot, Bhavna Rana

**Genre:** Garba

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Nat Bhairav | 0.9456 |
| 2 | Pahadi | 0.9453 |
| 3 | Purba | N/A |
| 4 | Bihag | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Nat Bhairav

**Top-1 Similarity:** 0.9456

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Nat Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 161 — Ghoomar
**Original Song ID:** GHOOMAR_001

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9272 |
| 2 | Megh | 0.9263 |
| 3 | Malkauns | N/A |
| 4 | Basant Mukhari | N/A |
| 5 | Bibhas | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9272

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 162 — Gangore/Gangaur
**Original Song ID:** GHOOMAR_002

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9523 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9458 |
| 3 | Adana | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9523

**Top-1 vs Top-2 Margin:** 0.0064

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 163 — Choumaso
**Original Song ID:** GHOOMAR_003

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman or Kalyan | 0.9322 |
| 2 | Bahar | 0.9316 |
| 3 | suddha kalyan | N/A |
| 4 | Purba | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Yaman or Kalyan

**Top-1 Similarity:** 0.9322

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Yaman or Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 164 — More Bole Re
**Original Song ID:** GHOOMAR_004

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9425 |
| 2 | Adana | 0.9379 |
| 3 | Malkauns | N/A |
| 4 | Basant Mukhari | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9425

**Top-1 vs Top-2 Margin:** 0.0045

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 165 — Podina
**Original Song ID:** GHOOMAR_005

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9418 |
| 2 | Malkauns | 0.9388 |
| 3 | Bhairav | N/A |
| 4 | Purba | N/A |
| 5 | Marva | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9418

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 166 — Talariya Magriya
**Original Song ID:** GHOOMAR_006

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9249 |
| 2 | Khamaj | 0.9217 |
| 3 | Bhairav | N/A |
| 4 | Marva | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9249

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 167 — Panihari Ji
**Original Song ID:** GHOOMAR_007

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9394 |
| 2 | Yaman or Kalyan | 0.9348 |
| 3 | Megh | N/A |
| 4 | Adana | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9394

**Top-1 vs Top-2 Margin:** 0.0046

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 168 — Roomal/Rumal
**Original Song ID:** GHOOMAR_008

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9330 |
| 2 | Yaman Kalyan | 0.9327 |
| 3 | Darbari | N/A |
| 4 | Khamaj | N/A |
| 5 | Sohni | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9330

**Top-1 vs Top-2 Margin:** 0.0004

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 169 — Gorband
**Original Song ID:** GHOOMAR_009

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9341 |
| 2 | Asavari | 0.9305 |
| 3 | suddha kalyan | N/A |
| 4 | Malkauns | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9341

**Top-1 vs Top-2 Margin:** 0.0036

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 170 — Banna Re Bagan Me Jhoola Dalya
**Original Song ID:** GHOOMAR_010

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9440 |
| 2 | Asavari | 0.9402 |
| 3 | Megh | N/A |
| 4 | Yaman or Kalyan | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9440

**Top-1 vs Top-2 Margin:** 0.0039

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 171 — Hivde Ro Haar
**Original Song ID:** GHOOMAR_011

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Barwa | 0.9375 |
| 2 | Khamaj | 0.9353 |
| 3 | Ahir Bhairav | N/A |
| 4 | Darbari | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Barwa

**Top-1 Similarity:** 0.9375

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Barwa was the highest-ranked candidate in the TRF reference space for this recording.

## 172 — Chirmi
**Original Song ID:** GHOOMAR_012

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9401 |
| 2 | Yaman or Kalyan | 0.9378 |
| 3 | Bhoopali | N/A |
| 4 | Adana | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9401

**Top-1 vs Top-2 Margin:** 0.0022

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 173 — Hichki (Aave Hichki)
**Original Song ID:** GHOOMAR_013

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9553 |
| 2 | Basant | 0.9499 |
| 3 | Bahar | N/A |
| 4 | Adana | N/A |
| 5 | Bibhas | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9553

**Top-1 vs Top-2 Margin:** 0.0054

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 174 — Dal Badal
**Original Song ID:** GHOOMAR_014

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9427 |
| 2 | Nat Bhairav | 0.9402 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Megh | N/A |
| 5 | Yaman or Kalyan | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9427

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 175 — Samdariyo
**Original Song ID:** GHOOMAR_015

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Miyan Ki Todi orTodi or Darbari Todi | 0.9456 |
| 2 | Bahar | 0.9454 |
| 3 | Durga | N/A |
| 4 | Madhuvanti | N/A |
| 5 | Megh | N/A |

**Top-1 Candidate:** Miyan Ki Todi orTodi or Darbari Todi

**Top-1 Similarity:** 0.9456

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Miyan Ki Todi orTodi or Darbari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 176 — Jeero
**Original Song ID:** GHOOMAR_016

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9399 |
| 2 | Yaman or Kalyan | 0.9396 |
| 3 | Adana | N/A |
| 4 | sarang | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9399

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 177 — Loonga Ro Batawo
**Original Song ID:** GHOOMAR_017

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9259 |
| 2 | Bhimpalasi | 0.9227 |
| 3 | Desi | N/A |
| 4 | Basant Mukhari | N/A |
| 5 | Megh | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9259

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 178 — Taran Ri Chunadi
**Original Song ID:** GHOOMAR_018

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9214 |
| 2 | Megh | 0.9207 |
| 3 | Jaunpuri | N/A |
| 4 | Pahadi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9214

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 179 — Run Jhun Baje Ghooghara
**Original Song ID:** GHOOMAR_019

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9360 |
| 2 | suddha kalyan | 0.9205 |
| 3 | Pahadi | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9360

**Top-1 vs Top-2 Margin:** 0.0156

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 180 — Kangasiyo
**Original Song ID:** GHOOMAR_020

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9296 |
| 2 | Desh | 0.9234 |
| 3 | Bageshree | N/A |
| 4 | Bhairav | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9296

**Top-1 vs Top-2 Margin:** 0.0062

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 181 — Dheere Chhal Re Panihari
**Original Song ID:** GHOOMAR_021

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9455 |
| 2 | Ahir Bhairav | 0.9445 |
| 3 | Desh | N/A |
| 4 | suddha kalyan | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9455

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 182 — Kesariya Ro Pag
**Original Song ID:** GHOOMAR_022

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9419 |
| 2 | Khamaj | 0.9373 |
| 3 | Sohni | N/A |
| 4 | Kedar | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9419

**Top-1 vs Top-2 Margin:** 0.0046

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 183 — Naina Ra Lobhi
**Original Song ID:** GHOOMAR_023

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Miyan Ki Todi orTodi or Darbari Todi | 0.9314 |
| 2 | Pahadi | 0.9305 |
| 3 | Multani | N/A |
| 4 | Rageshree | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Miyan Ki Todi orTodi or Darbari Todi

**Top-1 Similarity:** 0.9314

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Miyan Ki Todi orTodi or Darbari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 184 — Bajudan Ri Loom
**Original Song ID:** GHOOMAR_024

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9280 |
| 2 | Khamaj | 0.9270 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Megh Malhar | N/A |
| 5 | Barwa | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9280

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 185 — Lehariyo
**Original Song ID:** GHOOMAR_025

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9203 |
| 2 | Bhairav | 0.9193 |
| 3 | Kedar | N/A |
| 4 | Khamaj | N/A |
| 5 | Yaman Kalyan | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9203

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 186 — Ital Peetal
**Original Song ID:** GHOOMAR_026

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Durga | 0.9281 |
| 2 | Pahadi | 0.9254 |
| 3 | Darbari | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Durga

**Top-1 Similarity:** 0.9281

**Top-1 vs Top-2 Margin:** 0.0027

**Interpretation:**
Raag Durga was the highest-ranked candidate in the TRF reference space for this recording.

## 187 — Moriya
**Original Song ID:** GHOOMAR_027

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhatiyar | 0.9387 |
| 2 | Megh | 0.9352 |
| 3 | Lalit | N/A |
| 4 | Pahadi | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Bhatiyar

**Top-1 Similarity:** 0.9387

**Top-1 vs Top-2 Margin:** 0.0036

**Interpretation:**
Raag Bhatiyar was the highest-ranked candidate in the TRF reference space for this recording.

## 188 — Cham Cham Chake Chundari
**Original Song ID:** GHOOMAR_028

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Gujari Todi | 0.9231 |
| 2 | Madhuvanti | 0.9223 |
| 3 | Desh | N/A |
| 4 | Multani | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Gujari Todi

**Top-1 Similarity:** 0.9231

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Gujari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 189 — Hathelyan Re Beech
**Original Song ID:** GHOOMAR_029

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Barwa | 0.9361 |
| 2 | Khamaj | 0.9337 |
| 3 | Megh | N/A |
| 4 | Bahar | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Barwa

**Top-1 Similarity:** 0.9361

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Barwa was the highest-ranked candidate in the TRF reference space for this recording.

## 190 — Dar Lanjo
**Original Song ID:** GHOOMAR_030

**Artist:** Seema Mishra

**Genre:** Ghoomar

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9436 |
| 2 | Asavari | 0.9419 |
| 3 | Bhoopali | N/A |
| 4 | Adana | N/A |
| 5 | Megh Malhar | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9436

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 191 — Kati Khane Bole
**Original Song ID:** JHUMUR_001

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9283 |
| 2 | Bhatiyar | 0.9261 |
| 3 | Madhuvanti | N/A |
| 4 | Bhinna Shadja | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9283

**Top-1 vs Top-2 Margin:** 0.0022

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 192 — Ekadoshi Ratiya
**Original Song ID:** JHUMUR_0010

**Artist:** Zubeen Garg

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Lalit | 0.9387 |
| 2 | Bilaval | 0.9379 |
| 3 | Megh | N/A |
| 4 | Barwa | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Lalit

**Top-1 Similarity:** 0.9387

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Lalit was the highest-ranked candidate in the TRF reference space for this recording.

## 193 — Barire Simare
**Original Song ID:** JHUMUR_0011

**Artist:** Zubeen Garg

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9385 |
| 2 | Darbari | 0.9343 |
| 3 | Pahadi | N/A |
| 4 | suddha kalyan | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9385

**Top-1 vs Top-2 Margin:** 0.0041

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 194 — Mor Dhoni Go
**Original Song ID:** JHUMUR_0012

**Artist:** Zubeen Garg

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9293 |
| 2 | Madhuvanti | 0.9192 |
| 3 | Bilaval | N/A |
| 4 | Asavari | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9293

**Top-1 vs Top-2 Margin:** 0.0101

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 195 — Jhuli Jhule
**Original Song ID:** JHUMUR_0013

**Artist:** Zubeen Garg

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desi | 0.9251 |
| 2 | Madhuvanti | 0.9251 |
| 3 | Khamaj | N/A |
| 4 | Marva | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Desi

**Top-1 Similarity:** 0.9251

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Desi was the highest-ranked candidate in the TRF reference space for this recording.

## 196 — Kansiya
**Original Song ID:** JHUMUR_0014

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9427 |
| 2 | Kedar | 0.9418 |
| 3 | Asavari | N/A |
| 4 | Yaman Kalyan | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9427

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 197 — Jhatka
**Original Song ID:** JHUMUR_0015

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9381 |
| 2 | Sorath | 0.9366 |
| 3 | Megh | N/A |
| 4 | Barwa | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9381

**Top-1 vs Top-2 Margin:** 0.0015

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 198 — Gohona
**Original Song ID:** JHUMUR_0016

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9419 |
| 2 | Khamaj | 0.9387 |
| 3 | Darbari | N/A |
| 4 | Sohni | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9419

**Top-1 vs Top-2 Margin:** 0.0032

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 199 — Hariya
**Original Song ID:** JHUMUR_0017

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Kedar | 0.9342 |
| 2 | Darbari | 0.9303 |
| 3 | Bhoopali | N/A |
| 4 | Asavari | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Kedar

**Top-1 Similarity:** 0.9342

**Top-1 vs Top-2 Margin:** 0.0039

**Interpretation:**
Raag Kedar was the highest-ranked candidate in the TRF reference space for this recording.

## 200 — Koromoti
**Original Song ID:** JHUMUR_0018

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9395 |
| 2 | Bageshree | 0.9390 |
| 3 | Kedar | N/A |
| 4 | Adana | N/A |
| 5 | Bilaval | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9395

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 201 — Sonamoni
**Original Song ID:** JHUMUR_0019

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9368 |
| 2 | Desh | 0.9367 |
| 3 | Sorath | N/A |
| 4 | Bahar | N/A |
| 5 | Barwa | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9368

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 202 — Nistei
**Original Song ID:** JHUMUR_002

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9418 |
| 2 | Sohni | 0.9418 |
| 3 | Pahadi | N/A |
| 4 | Marva | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9418

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 203 — Chaliya Chhodi
**Original Song ID:** JHUMUR_0020

**Artist:** Amrit Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhoopali | 0.9277 |
| 2 | Malkauns | 0.9247 |
| 3 | Khamaj | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Bhoopali

**Top-1 Similarity:** 0.9277

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Bhoopali was the highest-ranked candidate in the TRF reference space for this recording.

## 204 — Prem Nadi Tire Tire
**Original Song ID:** JHUMUR_0021

**Artist:** Shaan Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9313 |
| 2 | Ahir Bhairav | 0.9311 |
| 3 | Adana | N/A |
| 4 | Sorath | N/A |
| 5 | Megh Malhar | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9313

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 205 — Deewana
**Original Song ID:** JHUMUR_0022

**Artist:** Shaan Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9413 |
| 2 | Bhimpalasi | 0.9378 |
| 3 | Sorath | N/A |
| 4 | Desh | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9413

**Top-1 vs Top-2 Margin:** 0.0035

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 206 — Koliye Juge Alai Doyal
**Original Song ID:** JHUMUR_0023

**Artist:** Navin Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Lalit | 0.9528 |
| 2 | Megh | 0.9527 |
| 3 | Adana | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Lalit

**Top-1 Similarity:** 0.9528

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Lalit was the highest-ranked candidate in the TRF reference space for this recording.

## 207 — Laal Pari
**Original Song ID:** JHUMUR_0024

**Artist:** Bejnath Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Madhuvanti | 0.9358 |
| 2 | Sorath | 0.9358 |
| 3 | Desh | N/A |
| 4 | Megh | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Madhuvanti

**Top-1 Similarity:** 0.9358

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Madhuvanti was the highest-ranked candidate in the TRF reference space for this recording.

## 208 — Tore Bina
**Original Song ID:** JHUMUR_0025

**Artist:** Bejnath Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9524 |
| 2 | Adana | 0.9521 |
| 3 | Bhoopali | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9524

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 209 — Koromake Akhora
**Original Song ID:** JHUMUR_0026

**Artist:** Bejnath Tanti

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9437 |
| 2 | Darbari | 0.9414 |
| 3 | Lalit | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9437

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 210 — The Jhumur Song
**Original Song ID:** JHUMUR_0027

**Artist:** Neel Akash

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Nat Bhairav | 0.9461 |
| 2 | Asavari | 0.9452 |
| 3 | Bhairav | N/A |
| 4 | Adana | N/A |
| 5 | Megh | N/A |

**Top-1 Candidate:** Nat Bhairav

**Top-1 Similarity:** 0.9461

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Nat Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 211 — Genda Phool
**Original Song ID:** JHUMUR_0028

**Artist:** Neel Akash, Jyotika Baruha

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9461 |
| 2 | Kedar | 0.9430 |
| 3 | Adana | N/A |
| 4 | Darbari | N/A |
| 5 | Bilaval | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9461

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 212 — Ja Ja Re Mini
**Original Song ID:** JHUMUR_0029

**Artist:** Neel Akash

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9344 |
| 2 | Asavari | 0.9334 |
| 3 | Pahadi | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9344

**Top-1 vs Top-2 Margin:** 0.0010

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 213 — Kuilu Kuilu
**Original Song ID:** JHUMUR_003

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9368 |
| 2 | Bhinna Shadja | 0.9347 |
| 3 | Bhairav | N/A |
| 4 | Basant | N/A |
| 5 | Basant Mukhari | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9368

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 214 — Goya
**Original Song ID:** JHUMUR_0030

**Artist:** Neel Akash

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9377 |
| 2 | Asavari | 0.9369 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9377

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 215 — Sajanigo
**Original Song ID:** JHUMUR_004

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desi | 0.9391 |
| 2 | Yaman or Kalyan | 0.9328 |
| 3 | Rageshree | N/A |
| 4 | Adana | N/A |
| 5 | Basant | N/A |

**Top-1 Candidate:** Desi

**Top-1 Similarity:** 0.9391

**Top-1 vs Top-2 Margin:** 0.0063

**Interpretation:**
Raag Desi was the highest-ranked candidate in the TRF reference space for this recording.

## 216 — Sol Sokhi
**Original Song ID:** JHUMUR_005

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Deepak | 0.9357 |
| 2 | Bhatiyar | 0.9334 |
| 3 | Marva | N/A |
| 4 | Lalit | N/A |
| 5 | Bhairav | N/A |

**Top-1 Candidate:** Deepak

**Top-1 Similarity:** 0.9357

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Deepak was the highest-ranked candidate in the TRF reference space for this recording.

## 217 — Aji Ei Suprabhate
**Original Song ID:** JHUMUR_006

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9240 |
| 2 | Bhairavi | 0.9224 |
| 3 | Kafi | N/A |
| 4 | Sorath | N/A |
| 5 | Marva | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9240

**Top-1 vs Top-2 Margin:** 0.0016

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 218 — Kale Gori Muke
**Original Song ID:** JHUMUR_007

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Nat Bhairav | 0.9320 |
| 2 | Malhar | 0.9309 |
| 3 | Malkauns | N/A |
| 4 | Basant | N/A |
| 5 | Basant Mukhari | N/A |

**Top-1 Candidate:** Nat Bhairav

**Top-1 Similarity:** 0.9320

**Top-1 vs Top-2 Margin:** 0.0011

**Interpretation:**
Raag Nat Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 219 — Tohe Gori Nawa
**Original Song ID:** JHUMUR_008

**Artist:** Dulal Manki

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9298 |
| 2 | Komal Bhairav | 0.9280 |
| 3 | Sohni | N/A |
| 4 | Malkauns | N/A |
| 5 | Bhinna Shadja | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9298

**Top-1 vs Top-2 Margin:** 0.0018

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 220 — Moi to Jabo Kam Jari
**Original Song ID:** JHUMUR_009

**Artist:** Zubeen Garg

**Genre:** Jhumur

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9310 |
| 2 | sarang | 0.9304 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Darbari | N/A |
| 5 | Sohni | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9310

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 221 — Tujhya Usala Lagal Kolha Ga
**Original Song ID:** LAVANI_001

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9381 |
| 2 | Bilaskhani Todi | 0.9374 |
| 3 | Basant | N/A |
| 4 | Puriya | N/A |
| 5 | Megh | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9381

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 222 — Padarawarti Jartaricha
**Original Song ID:** LAVANI_002

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9466 |
| 2 | Megh Malhar | 0.9454 |
| 3 | Asavari | N/A |
| 4 | Kedar | N/A |
| 5 | Nat Bhairav | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9466

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 223 — Solava Varis Dhokyacha
**Original Song ID:** LAVANI_003

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9312 |
| 2 | Bhairavi | 0.9306 |
| 3 | Adana | N/A |
| 4 | Komal Bhairav | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9312

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 224 — Kasa Kay Patil Bara Hay Ka
**Original Song ID:** LAVANI_004

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9427 |
| 2 | Bihag | 0.9410 |
| 3 | Khamaj | N/A |
| 4 | Bageshree | N/A |
| 5 | Bhimpalasi | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9427

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 225 — Naav Gaav Kashala Pustay
**Original Song ID:** LAVANI_005

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Nat Bhairav | 0.9527 |
| 2 | Rageshree | 0.9500 |
| 3 | Megh Malhar | N/A |
| 4 | Malhar | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Nat Bhairav

**Top-1 Similarity:** 0.9527

**Top-1 vs Top-2 Margin:** 0.0027

**Interpretation:**
Raag Nat Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 226 — Kalidar Kapuri Paan
**Original Song ID:** LAVANI_006

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9247 |
| 2 | Nat Bhairav | 0.9216 |
| 3 | Kafi | N/A |
| 4 | Basant | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9247

**Top-1 vs Top-2 Margin:** 0.0031

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 227 — Mala London Foreign Dakhwa
**Original Song ID:** LAVANI_007

**Artist:** Sulochana Chavan

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9369 |
| 2 | Bahar | 0.9315 |
| 3 | Yaman Kalyan | N/A |
| 4 | Bilaval | N/A |
| 5 | Yaman or Kalyan | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9369

**Top-1 vs Top-2 Margin:** 0.0053

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 228 — Disla Ga Bai Disla
**Original Song ID:** LAVANI_008

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Kedar | 0.9498 |
| 2 | Asavari | 0.9493 |
| 3 | Gauri | N/A |
| 4 | Marva | N/A |
| 5 | Megh Malhar | N/A |

**Top-1 Candidate:** Kedar

**Top-1 Similarity:** 0.9498

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Kedar was the highest-ranked candidate in the TRF reference space for this recording.

## 229 — Chabidar Chabi
**Original Song ID:** LAVANI_009

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | suddha kalyan | 0.9305 |
| 2 | Yaman or Kalyan | 0.9276 |
| 3 | Puriya | N/A |
| 4 | Pahadi | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** suddha kalyan

**Top-1 Similarity:** 0.9305

**Top-1 vs Top-2 Margin:** 0.0029

**Interpretation:**
Raag suddha kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 230 — Raya Chala Ghodyavarti Basu
**Original Song ID:** LAVANI_010

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9530 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9494 |
| 3 | Marva | N/A |
| 4 | Bihag | N/A |
| 5 | Lalit | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9530

**Top-1 vs Top-2 Margin:** 0.0036

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 231 — Pikala Jambhul Todu Naka
**Original Song ID:** LAVANI_011

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9584 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9557 |
| 3 | Bageshree | N/A |
| 4 | Rageshree | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9584

**Top-1 vs Top-2 Margin:** 0.0026

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 232 — Naki Doli Chhaan
**Original Song ID:** LAVANI_012

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Puriya | 0.9578 |
| 2 | Kedar | 0.9494 |
| 3 | Bageshree | N/A |
| 4 | Megh | N/A |
| 5 | Bihag | N/A |

**Top-1 Candidate:** Puriya

**Top-1 Similarity:** 0.9578

**Top-1 vs Top-2 Margin:** 0.0084

**Interpretation:**
Raag Puriya was the highest-ranked candidate in the TRF reference space for this recording.

## 233 — Aai Majhya Lagnachi
**Original Song ID:** LAVANI_013

**Artist:** Usha Mangeshkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malhar | 0.9455 |
| 2 | Marva | 0.9439 |
| 3 | Bhoopali | N/A |
| 4 | Durga | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Malhar

**Top-1 Similarity:** 0.9455

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Malhar was the highest-ranked candidate in the TRF reference space for this recording.

## 234 — Wajle Ki Bara
**Original Song ID:** LAVANI_014

**Artist:** Bela Shende

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9337 |
| 2 | Bhoopali | 0.9333 |
| 3 | Sohni | N/A |
| 4 | Puriya | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9337

**Top-1 vs Top-2 Margin:** 0.0004

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 235 — Apsara Aali
**Original Song ID:** LAVANI_015

**Artist:** Bela Shende

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant | 0.9305 |
| 2 | Sorath | 0.9297 |
| 3 | Bhimpalasi | N/A |
| 4 | Marva | N/A |
| 5 | Bhinna Shadja | N/A |

**Top-1 Candidate:** Basant

**Top-1 Similarity:** 0.9305

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Basant was the highest-ranked candidate in the TRF reference space for this recording.

## 236 — Kashi Mi Jau Mathurechya Bajari
**Original Song ID:** LAVANI_016

**Artist:** Bela Shende

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Shree | 0.9252 |
| 2 | Bahar | 0.9245 |
| 3 | Bhimpalasi | N/A |
| 4 | Sohni | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Shree

**Top-1 Similarity:** 0.9252

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Shree was the highest-ranked candidate in the TRF reference space for this recording.

## 237 — Kashyala Lavato
**Original Song ID:** LAVANI_017

**Artist:** Bela Shende

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Barwa | 0.9223 |
| 2 | Darbari | 0.9137 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Kafi | N/A |

**Top-1 Candidate:** Barwa

**Top-1 Similarity:** 0.9223

**Top-1 vs Top-2 Margin:** 0.0086

**Interpretation:**
Raag Barwa was the highest-ranked candidate in the TRF reference space for this recording.

## 238 — Chhatishi Kavlun Dhara
**Original Song ID:** LAVANI_018

**Artist:** Surekha Punekar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9357 |
| 2 | Madhuvanti | 0.9350 |
| 3 | Puriya | N/A |
| 4 | Khamaj | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9357

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 239 — Jhampar Majha Dheel
**Original Song ID:** LAVANI_019

**Artist:** Surekha Punekar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9265 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9250 |
| 3 | Puriya | N/A |
| 4 | Malkauns | N/A |
| 5 | Desi | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9265

**Top-1 vs Top-2 Margin:** 0.0015

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 240 — Themba Thembana Kelai
**Original Song ID:** LAVANI_020

**Artist:** Surekha Punekar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | suddha kalyan | 0.9305 |
| 2 | Yaman or Kalyan | 0.9276 |
| 3 | Puriya | N/A |
| 4 | Pahadi | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** suddha kalyan

**Top-1 Similarity:** 0.9305

**Top-1 vs Top-2 Margin:** 0.0029

**Interpretation:**
Raag suddha kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 241 — Sola Hazarat
**Original Song ID:** LAVANI_021

**Artist:** Uttara Kelkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9375 |
| 2 | Bilaskhani Todi | 0.9350 |
| 3 | Desh | N/A |
| 4 | Bageshree | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9375

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 242 — Hi Natrangi Naar
**Original Song ID:** LAVANI_022

**Artist:** Ravindra Sathe

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Puriya | 0.9424 |
| 2 | Marva | 0.9405 |
| 3 | Bageshree | N/A |
| 4 | Bhairav | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Puriya

**Top-1 Similarity:** 0.9424

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Puriya was the highest-ranked candidate in the TRF reference space for this recording.

## 243 — Saaman Avjad Majha
**Original Song ID:** LAVANI_023

**Artist:** Anand Shinde

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9265 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9219 |
| 3 | Bageshree | N/A |
| 4 | Malkauns | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9265

**Top-1 vs Top-2 Margin:** 0.0046

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 244 — Majha Taat Kar Aadhi
**Original Song ID:** LAVANI_024

**Artist:** Anand Shinde

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | suddha kalyan | 0.9290 |
| 2 | Yaman or Kalyan | 0.9283 |
| 3 | Puriya | N/A |
| 4 | Pahadi | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** suddha kalyan

**Top-1 Similarity:** 0.9290

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag suddha kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 245 — Bai Mi Patang Udvit Hote
**Original Song ID:** LAVANI_025

**Artist:** Asha Bhosle

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman or Kalyan | 0.9356 |
| 2 | Yaman Kalyan | 0.9351 |
| 3 | Gauri | N/A |
| 4 | Durga | N/A |
| 5 | Bahar | N/A |

**Top-1 Candidate:** Yaman or Kalyan

**Top-1 Similarity:** 0.9356

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Yaman or Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 246 — Bai Mazi Karangali Modali
**Original Song ID:** LAVANI_026

**Artist:** Asha Bhosle

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant Mukhari | 0.9551 |
| 2 | Puriya | 0.9550 |
| 3 | Bihag | N/A |
| 4 | Rageshree | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Basant Mukhari

**Top-1 Similarity:** 0.9551

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Basant Mukhari was the highest-ranked candidate in the TRF reference space for this recording.

## 247 — Raya Mala Pavsat Neu Naka
**Original Song ID:** LAVANI_027

**Artist:** Pushpa Pagdhare

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9522 |
| 2 | Bageshree | 0.9471 |
| 3 | Asavari | N/A |
| 4 | Purba | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9522

**Top-1 vs Top-2 Margin:** 0.0051

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 248 — Navi Navi Navari Mee
**Original Song ID:** LAVANI_028

**Artist:** Pushpa Pagdhare

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malhar | 0.9410 |
| 2 | Khamaj | 0.9365 |
| 3 | Desh | N/A |
| 4 | Marva | N/A |
| 5 | Kedar | N/A |

**Top-1 Candidate:** Malhar

**Top-1 Similarity:** 0.9410

**Top-1 vs Top-2 Margin:** 0.0045

**Interpretation:**
Raag Malhar was the highest-ranked candidate in the TRF reference space for this recording.

## 249 — Ugvali Shukrachi Chandni
**Original Song ID:** LAVANI_029

**Artist:** Aarti Anklikar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9206 |
| 2 | Desh | 0.9204 |
| 3 | Puriya | N/A |
| 4 | Darbari | N/A |
| 5 | Barwa | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9206

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 250 — Ghotala Ghotala Zala
**Original Song ID:** LAVANI_030

**Artist:** Vaishali Samant

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaskhani Todi | 0.9362 |
| 2 | Marva | 0.9360 |
| 3 | Desh | N/A |
| 4 | Bageshree | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Bilaskhani Todi

**Top-1 Similarity:** 0.9362

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Bilaskhani Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 251 — Raanamandi Ekatach Dultoy Naag
**Original Song ID:** LAVANI_031

**Artist:** Urmila Dhangkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9340 |
| 2 | Bhoopali | 0.9327 |
| 3 | Yaman or Kalyan | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9340

**Top-1 vs Top-2 Margin:** 0.0014

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 252 — Aase Wajwa Ki
**Original Song ID:** LAVANI_032

**Artist:** Bela Shende and chorus

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9426 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9421 |
| 3 | Nat Bhairav | N/A |
| 4 | Puriya | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9426

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 253 — Dolyavar Gogal Lava
**Original Song ID:** LAVANI_033

**Artist:** Various Artists

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9348 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9345 |
| 3 | Purba | N/A |
| 4 | sarang | N/A |
| 5 | Kedar | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9348

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 254 — Reshmachya Reghani
**Original Song ID:** LAVANI_034

**Artist:** Asha Bhosle

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9433 |
| 2 | Gauri | 0.9429 |
| 3 | Basant | N/A |
| 4 | Bahar | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9433

**Top-1 vs Top-2 Margin:** 0.0004

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 255 — Chala Jejurila Jau
**Original Song ID:** LAVANI_035

**Artist:** Uttara Kelkar

**Genre:** Lavani

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9391 |
| 2 | Desh | 0.9385 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Darbari | N/A |
| 5 | Bhairavi | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9391

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 256 — Kesariya Balam Aavo Ni
**Original Song ID:** MAAND_001

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9456 |
| 2 | Bahar | 0.9423 |
| 3 | Purba | N/A |
| 4 | Multani | N/A |
| 5 | Jaunpuri | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9456

**Top-1 vs Top-2 Margin:** 0.0034

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 257 — Moray Aaj Aaye Saiyan
**Original Song ID:** MAAND_002

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Hemant | 0.9179 |
| 2 | Bihag | 0.9174 |
| 3 | Bibhas | N/A |
| 4 | Lalit | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Hemant

**Top-1 Similarity:** 0.9179

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Hemant was the highest-ranked candidate in the TRF reference space for this recording.

## 258 — Moomal
**Original Song ID:** MAAND_003

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Madhuvanti | 0.9465 |
| 2 | Jaunpuri | 0.9453 |
| 3 | Multani | N/A |
| 4 | Desh | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Madhuvanti

**Top-1 Similarity:** 0.9465

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Madhuvanti was the highest-ranked candidate in the TRF reference space for this recording.

## 259 — Ghoomar
**Original Song ID:** MAAND_004

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9283 |
| 2 | Bilaskhani Todi | 0.9236 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9283

**Top-1 vs Top-2 Margin:** 0.0047

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 260 — Bai Sa Ra Beera
**Original Song ID:** MAAND_005

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9415 |
| 2 | Bahar | 0.9408 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Purba | N/A |
| 5 | Bhairav Bahar | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9415

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 261 — Jawai Balha Lagoji
**Original Song ID:** MAAND_006

**Artist:** Allah Jilai Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9220 |
| 2 | Bageshree | 0.9212 |
| 3 | Asavari | N/A |
| 4 | Gauri | N/A |
| 5 | Yaman or Kalyan | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9220

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 262 — Kesariya Balam Aavo Ni
**Original Song ID:** MAAND_007

**Artist:** Gavri Devi

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9233 |
| 2 | Madhuvanti | 0.9231 |
| 3 | Bahar | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | Sorath | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9233

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 263 — Kesariya Balam Awo Ni Padharo
**Original Song ID:** MAAND_008

**Artist:** Seema Mishra

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhatiyar | 0.9289 |
| 2 | Miyan Ki Todi orTodi or Darbari Todi | 0.9287 |
| 3 | Madhuvanti | N/A |
| 4 | Poorvi | N/A |
| 5 | Hemant | N/A |

**Top-1 Candidate:** Bhatiyar

**Top-1 Similarity:** 0.9289

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Bhatiyar was the highest-ranked candidate in the TRF reference space for this recording.

## 264 — Kesariya Ro Pag (Ghoomar)
**Original Song ID:** MAAND_009

**Artist:** Seema Mishra

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9444 |
| 2 | Ahir Bhairav | 0.9431 |
| 3 | Bhoopali | N/A |
| 4 | Asavari | N/A |
| 5 | Sohni | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9444

**Top-1 vs Top-2 Margin:** 0.0013

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 265 — Badila Yahi Rahiyo Saa
**Original Song ID:** MAAND_010

**Artist:** Banarsi Babu

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Gauri | 0.9537 |
| 2 | Bageshree | 0.9457 |
| 3 | Adana | N/A |
| 4 | Desh | N/A |
| 5 | Khamaj | N/A |

**Top-1 Candidate:** Gauri

**Top-1 Similarity:** 0.9537

**Top-1 vs Top-2 Margin:** 0.0080

**Interpretation:**
Raag Gauri was the highest-ranked candidate in the TRF reference space for this recording.

## 266 — Taniya Resham Ki
**Original Song ID:** MAAND_011

**Artist:** Banarsi Babu

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9454 |
| 2 | Khamaj | 0.9430 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9454

**Top-1 vs Top-2 Margin:** 0.0024

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 267 — Kad Janulee Sasre
**Original Song ID:** MAAND_012

**Artist:** Rehana Mirza

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9421 |
| 2 | Bhairavi | 0.9388 |
| 3 | Darbari | N/A |
| 4 | Bhairav | N/A |
| 5 | Bahar | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9421

**Top-1 vs Top-2 Margin:** 0.0034

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 268 — Le Dekh Nangi Melo
**Original Song ID:** MAAND_013

**Artist:** Rehana Mirza

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9477 |
| 2 | Malhar | 0.9466 |
| 3 | Megh Malhar | N/A |
| 4 | Purba | N/A |
| 5 | Yaman Kalyan | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9477

**Top-1 vs Top-2 Margin:** 0.0011

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 269 — The To Piyo Sadarudi
**Original Song ID:** MAAND_014

**Artist:** Rehana Mirza

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Rageshree | 0.9392 |
| 2 | Abhogi | 0.9385 |
| 3 | Bilaskhani Todi | N/A |
| 4 | Marva | N/A |
| 5 | Durga | N/A |

**Top-1 Candidate:** Rageshree

**Top-1 Similarity:** 0.9392

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Rageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 270 — Kesariya Balam (Desert Slide)
**Original Song ID:** MAAND_015

**Artist:** Anwar Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Miyan Ki Todi orTodi or Darbari Todi | 0.9267 |
| 2 | Malkauns | 0.9239 |
| 3 | Darbari | N/A |
| 4 | Sohni | N/A |
| 5 | Madhuvanti | N/A |

**Top-1 Candidate:** Miyan Ki Todi orTodi or Darbari Todi

**Top-1 Similarity:** 0.9267

**Top-1 vs Top-2 Margin:** 0.0029

**Interpretation:**
Raag Miyan Ki Todi orTodi or Darbari Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 271 — Kesariya Balam (DIFF 2015)
**Original Song ID:** MAAND_016

**Artist:** Anwar Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9186 |
| 2 | Madhuvanti | 0.9147 |
| 3 | Pahadi | N/A |
| 4 | Purba | N/A |
| 5 | Hemant | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9186

**Top-1 vs Top-2 Margin:** 0.0039

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 272 — Apne Desh Ri Aajadi
**Original Song ID:** MAAND_017

**Artist:** Anwar Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9351 |
| 2 | Basant | 0.9312 |
| 3 | Adana | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9351

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 273 — Kesariya Balam Aao Ni
**Original Song ID:** MAAND_018

**Artist:** Anwar Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9354 |
| 2 | Gauri | 0.9319 |
| 3 | Sohni | N/A |
| 4 | Khamaj | N/A |
| 5 | Basant | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9354

**Top-1 vs Top-2 Margin:** 0.0035

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 274 — Raag Maand Bandish
**Original Song ID:** MAAND_019

**Artist:** Mame Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9388 |
| 2 | Purba | 0.9388 |
| 3 | Sorath | N/A |
| 4 | Khamaj | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9388

**Top-1 vs Top-2 Margin:** 0.0000

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 275 — Rajasthani Folk Raag Maand
**Original Song ID:** MAAND_020

**Artist:** Mame Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9409 |
| 2 | Purba | 0.9397 |
| 3 | Sorath | N/A |
| 4 | Khamaj | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9409

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 276 — Chaudhary
**Original Song ID:** MAAND_021

**Artist:** Mame Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9324 |
| 2 | Desh | 0.9307 |
| 3 | Nat Bhairav | N/A |
| 4 | Bhairavi | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9324

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 277 — Chhap Tilak Sab Cheeni
**Original Song ID:** MAAND_022

**Artist:** Kirti Sagathia

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9323 |
| 2 | Bahar | 0.9323 |
| 3 | Darbari | N/A |
| 4 | Nat Bhairav | N/A |
| 5 | suddha kalyan | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9323

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 278 — Balam
**Original Song ID:** MAAND_023

**Artist:** Swaroop Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9495 |
| 2 | Asavari | 0.9451 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9495

**Top-1 vs Top-2 Margin:** 0.0044

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 279 — Kuraja Mahra Bhanvr Manae Lav
**Original Song ID:** MAAND_024

**Artist:** Swaroop Khan Sura

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9147 |
| 2 | Ahir Bhairav | 0.9136 |
| 3 | Barwa | N/A |
| 4 | Bhairavi | N/A |
| 5 | Bibhas | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9147

**Top-1 vs Top-2 Margin:** 0.0011

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 280 — Kurja Mhari Bhanwar Milade
**Original Song ID:** MAAND_025

**Artist:** Sikander Khan Baru

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9313 |
| 2 | Adana | 0.9296 |
| 3 | Bahar | N/A |
| 4 | Pahadi | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9313

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 281 — Darude Ro Rog Lagiyo
**Original Song ID:** MAAND_026

**Artist:** Sikander Khan Baru

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9131 |
| 2 | Puriya | 0.9098 |
| 3 | Darbari | N/A |
| 4 | Asavari | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9131

**Top-1 vs Top-2 Margin:** 0.0033

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 282 — Kesariya Balam Padharo
**Original Song ID:** MAAND_027

**Artist:** Bhungar Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Komal Bhairav | 0.9439 |
| 2 | Bhoopali | 0.9401 |
| 3 | Bhimpalasi | N/A |
| 4 | Khamaj | N/A |
| 5 | Purba | N/A |

**Top-1 Candidate:** Komal Bhairav

**Top-1 Similarity:** 0.9439

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Komal Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 283 — Mohe Aai Na Jag Ki Laaj
**Original Song ID:** MAAND_028

**Artist:** Kheta Khan, Fakira Khan Bhadresh

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Kafi | 0.9416 |
| 2 | Pahadi | 0.9397 |
| 3 | Desh | N/A |
| 4 | Bilaskhani Todi | N/A |
| 5 | Malhar | N/A |

**Top-1 Candidate:** Kafi

**Top-1 Similarity:** 0.9416

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Kafi was the highest-ranked candidate in the TRF reference space for this recording.

## 284 — Kalali
**Original Song ID:** MAAND_029

**Artist:** Latif Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Malkauns | 0.9186 |
| 2 | Bhairavi | 0.9169 |
| 3 | Megh | N/A |
| 4 | Desh | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Malkauns

**Top-1 Similarity:** 0.9186

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Malkauns was the highest-ranked candidate in the TRF reference space for this recording.

## 285 — Kesariya Balam
**Original Song ID:** MAAND_030

**Artist:** Raitila (Group)

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9278 |
| 2 | Sorath | 0.9263 |
| 3 | Purba | N/A |
| 4 | Malkauns | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9278

**Top-1 vs Top-2 Margin:** 0.0014

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 286 — Joori Thaane Na Maare Raad
**Original Song ID:** MAAND_031

**Artist:** Jameela Bai & Kulsum Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Yaman Kalyan | 0.9218 |
| 2 | Bhimpalasi | 0.9212 |
| 3 | Bilaskhani Todi | N/A |
| 4 | Lalit | N/A |
| 5 | Basant | N/A |

**Top-1 Candidate:** Yaman Kalyan

**Top-1 Similarity:** 0.9218

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Yaman Kalyan was the highest-ranked candidate in the TRF reference space for this recording.

## 287 — Alija Dhola Ekladi
**Original Song ID:** MAAND_032

**Artist:** Jameela Bai & Kulsum Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9228 |
| 2 | Bhimpalasi | 0.9190 |
| 3 | Yaman Kalyan | N/A |
| 4 | Bhairavi | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9228

**Top-1 vs Top-2 Margin:** 0.0037

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 288 — Mumal
**Original Song ID:** MAAND_033

**Artist:** Jameela Bai & Kulsum Bai

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9342 |
| 2 | Yaman Kalyan | 0.9331 |
| 3 | Bhimpalasi | N/A |
| 4 | Bhatiyar | N/A |
| 5 | Bilaskhani Todi | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9342

**Top-1 vs Top-2 Margin:** 0.0012

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 289 — Mhari Sajni Ae
**Original Song ID:** MAAND_034

**Artist:** Nand Lal Bhat

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9439 |
| 2 | Bhairavi | 0.9420 |
| 3 | Yaman Kalyan | N/A |
| 4 | Puriya | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9439

**Top-1 vs Top-2 Margin:** 0.0019

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 290 — Rajwadi Mehfil
**Original Song ID:** MAAND_035

**Artist:** Nand Lal Bhat

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9365 |
| 2 | Marva | 0.9363 |
| 3 | Megh | N/A |
| 4 | Bhairavi | N/A |
| 5 | Basant | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9365

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 291 — Thane Samjhaye Sakha Sa
**Original Song ID:** MAAND_036

**Artist:** Munshi Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9420 |
| 2 | Puriya | 0.9395 |
| 3 | Bibhas | N/A |
| 4 | Adana | N/A |
| 5 | Basant | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9420

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 292 — Padharo Mhare Mehal
**Original Song ID:** MAAND_037

**Artist:** Munshi Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhupal Todi | 0.9445 |
| 2 | Ramkali | 0.9417 |
| 3 | Komal Bhairav | N/A |
| 4 | Bibhas | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bhupal Todi

**Top-1 Similarity:** 0.9445

**Top-1 vs Top-2 Margin:** 0.0027

**Interpretation:**
Raag Bhupal Todi was the highest-ranked candidate in the TRF reference space for this recording.

## 293 — Daru Dakha Ro
**Original Song ID:** MAAND_038

**Artist:** Munshi Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Gauri | 0.9424 |
| 2 | Alhaiya Bilaval | 0.9399 |
| 3 | Malhar | N/A |
| 4 | Khamaj | N/A |
| 5 | Kafi | N/A |

**Top-1 Candidate:** Gauri

**Top-1 Similarity:** 0.9424

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Gauri was the highest-ranked candidate in the TRF reference space for this recording.

## 294 — Mahara Sajan Base
**Original Song ID:** MAAND_039

**Artist:** Munshi Khan

**Genre:** Maand

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Purba | 0.9428 |
| 2 | Desh | 0.9375 |
| 3 | Yaman Kalyan | N/A |
| 4 | Adana | N/A |
| 5 | Bhatiyar | N/A |

**Top-1 Candidate:** Purba

**Top-1 Similarity:** 0.9428

**Top-1 vs Top-2 Margin:** 0.0053

**Interpretation:**
Raag Purba was the highest-ranked candidate in the TRF reference space for this recording.

## 295 — Nabu Ki Sambalpur
**Original Song ID:** SAMBALPURI_001

**Artist:** Human Sagar

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Ahir Bhairav | 0.9407 |
| 2 | Malkauns | 0.9363 |
| 3 | Pahadi | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Ahir Bhairav

**Top-1 Similarity:** 0.9407

**Top-1 vs Top-2 Margin:** 0.0044

**Interpretation:**
Raag Ahir Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## 296 — Tor Lagi Marijibi
**Original Song ID:** SAMBALPURI_002

**Artist:** Human Sagar

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9375 |
| 2 | Ahir Bhairav | 0.9366 |
| 3 | Sorath | N/A |
| 4 | Darbari | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9375

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 297 — Sacha Lover
**Original Song ID:** SAMBALPURI_003

**Artist:** Human Sagar

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant | 0.9391 |
| 2 | Darbari | 0.9370 |
| 3 | Asavari | N/A |
| 4 | Sorath | N/A |
| 5 | Marva | N/A |

**Top-1 Candidate:** Basant

**Top-1 Similarity:** 0.9391

**Top-1 vs Top-2 Margin:** 0.0021

**Interpretation:**
Raag Basant was the highest-ranked candidate in the TRF reference space for this recording.

## 298 — Pardesia Raja
**Original Song ID:** SAMBALPURI_004

**Artist:** Human Sagar

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9508 |
| 2 | Darbari | 0.9491 |
| 3 | Bageshree | N/A |
| 4 | Desh | N/A |
| 5 | Miyan Ki Todi orTodi or Darbari Todi | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9508

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 299 — Lal Sadhi
**Original Song ID:** SAMBALPURI_005

**Artist:** Human Sagar

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9279 |
| 2 | Desh | 0.9254 |
| 3 | Bageshree | N/A |
| 4 | Bhairavi | N/A |
| 5 | Kedar | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9279

**Top-1 vs Top-2 Margin:** 0.0025

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 300 — Rasia
**Original Song ID:** SAMBALPURI_006

**Artist:** Mantu Chhuria

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bilaval | 0.9351 |
| 2 | Bageshree | 0.9276 |
| 3 | Bhimpalasi | N/A |
| 4 | Asavari | N/A |
| 5 | Desh | N/A |

**Top-1 Candidate:** Bilaval

**Top-1 Similarity:** 0.9351

**Top-1 vs Top-2 Margin:** 0.0074

**Interpretation:**
Raag Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 301 — A Mo Dil
**Original Song ID:** SAMBALPURI_007

**Artist:** Mantu Chhuria

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9364 |
| 2 | Bageshree | 0.9330 |
| 3 | Adana | N/A |
| 4 | Asavari | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9364

**Top-1 vs Top-2 Margin:** 0.0034

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 302 — Hae Go
**Original Song ID:** SAMBALPURI_008

**Artist:** Mantu Chhuria

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9347 |
| 2 | Megh | 0.9345 |
| 3 | Sorath | N/A |
| 4 | Bilaval | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9347

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 303 — Aukaat
**Original Song ID:** SAMBALPURI_009

**Artist:** Mantu Chhuria

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9504 |
| 2 | Bhatiyar | 0.9495 |
| 3 | Sohni | N/A |
| 4 | Bhimpalasi | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9504

**Top-1 vs Top-2 Margin:** 0.0008

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 304 — Rani Guri
**Original Song ID:** SAMBALPURI_010

**Artist:** Mantu Chhuria

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9397 |
| 2 | Puriya | 0.9367 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Adana | N/A |
| 5 | Bhoopali | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9397

**Top-1 vs Top-2 Margin:** 0.0030

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 305 — Gidja Gija Re
**Original Song ID:** SAMBALPURI_011

**Artist:** Asima Panda

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9379 |
| 2 | Bageshree | 0.9376 |
| 3 | Bilaval | N/A |
| 4 | Adana | N/A |
| 5 | Abhogi | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9379

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 306 — Tui Bewafa
**Original Song ID:** SAMBALPURI_012

**Artist:** Asima Panda

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Desh | 0.9439 |
| 2 | Adana | 0.9402 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Sorath | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Desh

**Top-1 Similarity:** 0.9439

**Top-1 vs Top-2 Margin:** 0.0037

**Interpretation:**
Raag Desh was the highest-ranked candidate in the TRF reference space for this recording.

## 307 — Ramagaru
**Original Song ID:** SAMBALPURI_013

**Artist:** Asima Panda

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9276 |
| 2 | Desh | 0.9275 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9276

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 308 — Deewana Banalu Rani
**Original Song ID:** SAMBALPURI_014

**Artist:** Asima Panda

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9338 |
| 2 | Megh | 0.9335 |
| 3 | Rageshree | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Adana | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9338

**Top-1 vs Top-2 Margin:** 0.0002

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 309 — SambalPuri Saiyaan
**Original Song ID:** SAMBALPURI_015

**Artist:** Asima Panda

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Sorath | 0.9390 |
| 2 | Desh | 0.9374 |
| 3 | Adana | N/A |
| 4 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Sorath

**Top-1 Similarity:** 0.9390

**Top-1 vs Top-2 Margin:** 0.0016

**Interpretation:**
Raag Sorath was the highest-ranked candidate in the TRF reference space for this recording.

## 310 — Anta Halei Halei
**Original Song ID:** SAMBALPURI_016

**Artist:** Ruku Suna

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Alhaiya Bilaval | 0.9476 |
| 2 | Bhairavi | 0.9467 |
| 3 | Pahadi | N/A |
| 4 | Darbari | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Alhaiya Bilaval

**Top-1 Similarity:** 0.9476

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Alhaiya Bilaval was the highest-ranked candidate in the TRF reference space for this recording.

## 311 — Chandir Paenjhal
**Original Song ID:** SAMBALPURI_017

**Artist:** Ruku Suna

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9446 |
| 2 | Asavari | 0.9443 |
| 3 | Desh | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9446

**Top-1 vs Top-2 Margin:** 0.0003

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 312 — Prem Paga
**Original Song ID:** SAMBALPURI_018

**Artist:** Ruku Suna

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Darbari | 0.9492 |
| 2 | Asavari | 0.9469 |
| 3 | Adana | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Darbari

**Top-1 Similarity:** 0.9492

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Darbari was the highest-ranked candidate in the TRF reference space for this recording.

## 313 — Rasa Mandara Mali Kahanra
**Original Song ID:** SAMBALPURI_019

**Artist:** Ruku Suna

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Megh | 0.9427 |
| 2 | Hindol | 0.9419 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Adana | N/A |
| 5 | Nat Bhairav | N/A |

**Top-1 Candidate:** Megh

**Top-1 Similarity:** 0.9427

**Top-1 vs Top-2 Margin:** 0.0007

**Interpretation:**
Raag Megh was the highest-ranked candidate in the TRF reference space for this recording.

## 314 — A Labanga Lata
**Original Song ID:** SAMBALPURI_020

**Artist:** Ruku Suna

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairavi | 0.9297 |
| 2 | Bhoopali | 0.9291 |
| 3 | Asavari | N/A |
| 4 | Abhogi | N/A |
| 5 | Pahadi | N/A |

**Top-1 Candidate:** Bhairavi

**Top-1 Similarity:** 0.9297

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Bhairavi was the highest-ranked candidate in the TRF reference space for this recording.

## 315 — Badnam
**Original Song ID:** SAMBALPURI_021

**Artist:** Prakash Jal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Rageshree | 0.9225 |
| 2 | Kedar | 0.9185 |
| 3 | Pahadi | N/A |
| 4 | Ahir Bhairav | N/A |
| 5 | Yaman or Kalyan | N/A |

**Top-1 Candidate:** Rageshree

**Top-1 Similarity:** 0.9225

**Top-1 vs Top-2 Margin:** 0.0040

**Interpretation:**
Raag Rageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 316 — Dil Bhitre Lekhideli
**Original Song ID:** SAMBALPURI_022

**Artist:** Prakash Jal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Pahadi | 0.9342 |
| 2 | Ahir Bhairav | 0.9319 |
| 3 | N/A | N/A |
| 4 | N/A | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Pahadi

**Top-1 Similarity:** 0.9342

**Top-1 vs Top-2 Margin:** 0.0023

**Interpretation:**
Raag Pahadi was the highest-ranked candidate in the TRF reference space for this recording.

## 317 — Bewafa Mita
**Original Song ID:** SAMBALPURI_023

**Artist:** Prakash Jal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Adana | 0.9353 |
| 2 | Durga | 0.9340 |
| 3 | Bhoopali | N/A |
| 4 | Pahadi | N/A |
| 5 | Marva | N/A |

**Top-1 Candidate:** Adana

**Top-1 Similarity:** 0.9353

**Top-1 vs Top-2 Margin:** 0.0013

**Interpretation:**
Raag Adana was the highest-ranked candidate in the TRF reference space for this recording.

## 318 — Bheti Aibu Sajani Kalekale
**Original Song ID:** SAMBALPURI_024

**Artist:** Prakash Jal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Pahadi | 0.9374 |
| 2 | Nat Bhairav | 0.9368 |
| 3 | Miyan Ki Todi orTodi or Darbari Todi | N/A |
| 4 | Bihag | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Pahadi

**Top-1 Similarity:** 0.9374

**Top-1 vs Top-2 Margin:** 0.0006

**Interpretation:**
Raag Pahadi was the highest-ranked candidate in the TRF reference space for this recording.

## 319 — Le Le Baula
**Original Song ID:** SAMBALPURI_025

**Artist:** Antara Chakrabarty

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Asavari | 0.9484 |
| 2 | Adana | 0.9467 |
| 3 | Alhaiya Bilaval | N/A |
| 4 | Puriya | N/A |
| 5 | Ahir Bhairav | N/A |

**Top-1 Candidate:** Asavari

**Top-1 Similarity:** 0.9484

**Top-1 vs Top-2 Margin:** 0.0017

**Interpretation:**
Raag Asavari was the highest-ranked candidate in the TRF reference space for this recording.

## 320 — Chahanewali
**Original Song ID:** SAMBALPURI_026

**Artist:** Antara Chakrabarty

**Genre:** Sambalpuri

**Status:** Excluded

NO PREDICTION — insufficient data for a complete 60-second sequence.

## 321 — Feel My Love
**Original Song ID:** SAMBALPURI_027

**Artist:** Antara Chakrabarty

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9430 |
| 2 | Madhuvanti | 0.9425 |
| 3 | Bhimpalasi | N/A |
| 4 | Desh | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9430

**Top-1 vs Top-2 Margin:** 0.0005

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 322 — Kulfirani Chocobar
**Original Song ID:** SAMBALPURI_028

**Artist:** Ira Mohanty

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Marva | 0.9496 |
| 2 | Megh | 0.9429 |
| 3 | Asavari | N/A |
| 4 | Alhaiya Bilaval | N/A |
| 5 | N/A | N/A |

**Top-1 Candidate:** Marva

**Top-1 Similarity:** 0.9496

**Top-1 vs Top-2 Margin:** 0.0067

**Interpretation:**
Raag Marva was the highest-ranked candidate in the TRF reference space for this recording.

## 323 — Jhupa Jhupa Baul Jhupa
**Original Song ID:** SAMBALPURI_029

**Artist:** Ira Mohanty

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Khamaj | 0.9419 |
| 2 | Bageshree | 0.9366 |
| 3 | Asavari | N/A |
| 4 | Adana | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Khamaj

**Top-1 Similarity:** 0.9419

**Top-1 vs Top-2 Margin:** 0.0053

**Interpretation:**
Raag Khamaj was the highest-ranked candidate in the TRF reference space for this recording.

## 324 — Sundri Janha
**Original Song ID:** SAMBALPURI_030

**Artist:** Ira Mohanty

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bageshree | 0.9371 |
| 2 | Asavari | 0.9356 |
| 3 | Ahir Bhairav | N/A |
| 4 | Kedar | N/A |
| 5 | Darbari | N/A |

**Top-1 Candidate:** Bageshree

**Top-1 Similarity:** 0.9371

**Top-1 vs Top-2 Margin:** 0.0015

**Interpretation:**
Raag Bageshree was the highest-ranked candidate in the TRF reference space for this recording.

## 325 — Rangabati
**Original Song ID:** SAMBALPURI_031

**Artist:** Jitendra Haripal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bahar | 0.9151 |
| 2 | Kedar | 0.9113 |
| 3 | Adana | N/A |
| 4 | Rageshree | N/A |
| 5 | Bhairav | N/A |

**Top-1 Candidate:** Bahar

**Top-1 Similarity:** 0.9151

**Top-1 vs Top-2 Margin:** 0.0038

**Interpretation:**
Raag Bahar was the highest-ranked candidate in the TRF reference space for this recording.

## 326 — Aha Mor Galar Malire
**Original Song ID:** SAMBALPURI_032

**Artist:** Jitendra Haripal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Basant | 0.9320 |
| 2 | Marva | 0.9267 |
| 3 | Ahir Bhairav | N/A |
| 4 | Sorath | N/A |
| 5 | Malkauns | N/A |

**Top-1 Candidate:** Basant

**Top-1 Similarity:** 0.9320

**Top-1 vs Top-2 Margin:** 0.0052

**Interpretation:**
Raag Basant was the highest-ranked candidate in the TRF reference space for this recording.

## 327 — Din Gani Gani
**Original Song ID:** SAMBALPURI_033

**Artist:** Jitendra Haripal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Shree | 0.9340 |
| 2 | Alhaiya Bilaval | 0.9339 |
| 3 | Khamaj | N/A |
| 4 | Desi | N/A |
| 5 | Asavari | N/A |

**Top-1 Candidate:** Shree

**Top-1 Similarity:** 0.9340

**Top-1 vs Top-2 Margin:** 0.0001

**Interpretation:**
Raag Shree was the highest-ranked candidate in the TRF reference space for this recording.

## 328 — Namo Namo Ago
**Original Song ID:** SAMBALPURI_034

**Artist:** Jitendra Haripal

**Genre:** Sambalpuri

**Status:** Successfully classified

### Retrieved Raag Candidates

| Rank | Raag | Cosine Similarity |
|---|---|---:|
| 1 | Bhairav | 0.9314 |
| 2 | Kedar | 0.9306 |
| 3 | Rageshree | N/A |
| 4 | Bhoopali | N/A |
| 5 | Puriya | N/A |

**Top-1 Candidate:** Bhairav

**Top-1 Similarity:** 0.9314

**Top-1 vs Top-2 Margin:** 0.0009

**Interpretation:**
Raag Bhairav was the highest-ranked candidate in the TRF reference space for this recording.

## PART 3 — EXCLUDED RECORDING

# Excluded Recording
- **Original Song ID:** SAMBALPURI_026
- **Song Name:** Chahanewali
- **Artist:** Antara Chakrabarty
- **Genre:** Sambalpuri
- **Reason for exclusion:** Insufficient 5-second segments.

No Raag prediction was generated because the recording did not contain enough contiguous valid segments to construct the required 60-second representation.

## PART 4 — GENRE ORGANIZATION

## Baul
- Total recordings: 30
- Successfully classified: 30
- Excluded: 0
- Most frequently retrieved Raag: Yaman Kalyan
- Second most frequent Raag: Bhairavi
- Third most frequent Raag: Purba
- Average Top-1 similarity: 0.9329
- Average Top-1/Top-2 margin: 0.0347

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| BAUL_001 | Agun Pani | Purna Das Baul | Madhuvanti | 0.9390 | 0.0008 |
| BAUL_002 | Ja Lolite Anga Kushum Tule | Purna Das Baul | Abhogi | 0.9225 | 0.0025 |
| BAUL_003 | Kalkatou Mon Heshekeley | Purna Das Baul | Yaman Kalyan | 0.9422 | 0.0029 |
| BAUL_004 | Ananda Lahari | Purna Das Baul | Marva | 0.9303 | 0.0039 |
| BAUL_005 | Amar Aei Rudir Dhara | Purna Das Baul | Bhairavi | 0.9324 | 0.0031 |
| BAUL_006 | Hari Balo Re | Purna Das Baul | Yaman Kalyan | 0.9325 | 0.0033 |
| BAUL_007 | Gour Lila Bazare | Purna Das Baul | Yaman Kalyan | 0.9373 | 0.0059 |
| BAUL_008 | Tripti | Purna Das Baul | Malkauns | 0.8812 | 0.0002 |
| BAUL_009 | Kon Ba Zharir | Purna Das Baul | Barwa | 0.9362 | 0.0012 |
| BAUL_010 | Sonogo Aian Dada | Purna Das Baul | Yaman Kalyan | 0.9457 | 0.0040 |
| BAUL_011 | Chor Poreche Babur Bagane | Purna Das Baul | Bhairavi | 0.9281 | 0.0076 |
| BAUL_012 | Golemale Gole Male | Purna Das Baul | Khamaj | 0.9357 | 0.0016 |
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Purna Das Baul | Puriya | 0.9328 | 0.0052 |
| BAUL_014 | Dekhechhi Rup-Sagare | Purna Das Baul | Bhairavi | 0.9367 | 0.0049 |
| BAUL_015 | Aami Kothay Pabo Tare | Purna Das Baul | Abhogi | 0.9181 | 0.0009 |
| BAUL_016 | Shesher Deene Sejon Bine | Purna Das Baul | Sorath | 0.9355 | 0.0051 |
| BAUL_017 | Milon Hobe Koto Dine | Farida Parveen | Bhimpalasi | 0.9335 | 0.0017 |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Farida Parveen | Abhogi | 0.9294 | 0.0005 |
| BAUL_019 | Aami Ek Pagal Baul | Purna Das Baul | Bhairavi | 0.9208 | 0.0001 |
| BAUL_020 | Jemon Beni Temni Rabey | Purna Das Baul | Bhairav | 0.9400 | 0.0043 |
| BAUL_021 | Prem Koraa Ki Jwalaa | Purna Das Baul | Purba | 0.9417 | 0.0084 |
| BAUL_022 | Emon Mon Byabsaa | Purna Das Baul | Durga | 0.9488 | 0.0040 |
| BAUL_023 | Manush Bhajo Manusheyi Pujo | Purna Das Baul | Purba | 0.9455 | 0.9455 |
| BAUL_024 | Dil Dariyar Majhete | Purna Das Baul | Purba | 0.9334 | 0.0088 |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Purna Das Baul | Purba | 0.9353 | 0.0059 |
| BAUL_026 | Sokhi Jamuna Jol | Purna Das Baul | Malkauns | 0.9320 | 0.0005 |
| BAUL_027 | Pre Loye Jao Amay | Farida Parveen | Malkauns | 0.9434 | 0.0032 |
| BAUL_028 | Somoy Gele Sadhon Hobena | Farida Parveen | Nat Bhairav | 0.9180 | 0.0009 |
| BAUL_029 | Aar Amare Marishne Maa | Farida Parveen | Miyan Ki Todi orTodi or Darbari Todi | 0.9414 | 0.0020 |
| BAUL_030 | Dhonno Dhonno Boli Tare | Farida Parveen | Bahar | 0.9370 | 0.0031 |

## Bhangra
- Total recordings: 32
- Successfully classified: 32
- Excluded: 0
- Most frequently retrieved Raag: Darbari
- Second most frequent Raag: Adana
- Third most frequent Raag: Gujari Todi
- Average Top-1 similarity: 0.9335
- Average Top-1/Top-2 margin: 0.0036

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| BHANGRA_001 | Tera Yaar Bolda | Surjit Bindrakhia | Bahar | 0.9212 | 0.0001 |
| BHANGRA_002 | Jatt Di Pasand | Surjit Bindrakhia | Bhoopali | 0.9307 | 0.0035 |
| BHANGRA_003 | Dupatta Tera Satrang Da | Surjit Bindrakhia | Darbari | 0.9340 | 0.0000 |
| BHANGRA_004 | Wang Wargi Kudi | Surjit Bindrakhia | Darbari | 0.9286 | 0.0043 |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Surjit Bindrakhia | Yaman or Kalyan | 0.9279 | 0.0041 |
| BHANGRA_006 | Munda Ki Mangda | Surjit Bindrakhia | Adana | 0.9311 | 0.0029 |
| BHANGRA_007 | Tunak Tunak Tun | Daler Mehndi | Megh | 0.9487 | 0.0127 |
| BHANGRA_008 | Bolo Ta Ra Ra | Daler Mehndi | Desh | 0.9417 | 0.0032 |
| BHANGRA_009 | Ho Jayegi Balle Balle | Daler Mehndi | Darbari | 0.9468 | 0.0071 |
| BHANGRA_010 | Dardi Rab Rab Kardi | Daler Mehndi | Darbari | 0.9320 | 0.0045 |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Daler Mehndi | Adana | 0.9344 | 0.0046 |
| BHANGRA_012 | Na Na Na Na Na Re | Daler Mehndi | Adana | 0.9397 | 0.0015 |
| BHANGRA_013 | Pao Bhangra | Balwinder Safri | Bhairavi | 0.9151 | 0.0011 |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Balwinder Safri | Gujari Todi | 0.9384 | 0.0081 |
| BHANGRA_015 | Sariyan Toon Sohni | Balwinder Safri | Alhaiya Bilaval | 0.9291 | 0.0005 |
| BHANGRA_016 | Mitran Ne Mundri Te | Balwinder Safri | Darbari | 0.9400 | 0.0006 |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Balwinder Safri | Gujari Todi | 0.9387 | 0.0054 |
| BHANGRA_018 | Tutak Tutak Tutiya | Malkit Singh (Golden Star) | Darbari | 0.9297 | 0.0002 |
| BHANGRA_019 | Kurri Garam Jayee | Malkit Singh (Golden Star) | Rageshree | 0.9322 | 0.0003 |
| BHANGRA_020 | Chal Hun | Malkit Singh (Golden Star) | Shree | 0.9341 | 0.0014 |
| BHANGRA_021 | Kinni Sohni | Malkit Singh (Golden Star) | Darbari | 0.9421 | 0.0063 |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Apna Sangeet | Bilaval | 0.9433 | 0.0009 |
| BHANGRA_023 | Bhaina Nachiyaan | Apna Sangeet | Alhaiya Bilaval | 0.9301 | 0.0049 |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Apna Sangeet | Bilaval | 0.9226 | 0.0081 |
| BHANGRA_025 | Soho Road Utey | Apna Sangeet | Malkauns | 0.9347 | 0.0026 |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Kuldeep Manak | Darbari | 0.9309 | 0.0065 |
| BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Kuldeep Manak | Darbari | 0.9342 | 0.0030 |
| BHANGRA_028 | Jugni | Kuldeep Manak | Yaman Kalyan | 0.9351 | 0.0043 |
| BHANGRA_029 | Putt Jattan De | Surinder Shinda | Abhogi | 0.9170 | 0.0002 |
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Surinder Shinda | suddha kalyan | 0.9264 | 0.0035 |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Surinder Shinda | Rageshree | 0.9379 | 0.0090 |
| BHANGRA_032 | Jeona Morh | Surinder Shinda | Basant Mukhari | 0.9448 | 0.0007 |

## Bihu
- Total recordings: 37
- Successfully classified: 37
- Excluded: 0
- Most frequently retrieved Raag: Bhoopali
- Second most frequent Raag: Desh
- Third most frequent Raag: Ahir Bhairav
- Average Top-1 similarity: 0.9352
- Average Top-1/Top-2 margin: 0.0020

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| BIHU_001 | Morom Logai Logai | Zubeen Garg | Sohni | 0.9355 | 0.0021 |
| BIHU_002 | Toi Nagini Ne | Zubeen Garg | Asavari | 0.9280 | 0.0009 |
| BIHU_003 | Kokalote Tongali | Zubeen Garg | Hindol | 0.9402 | 0.0002 |
| BIHU_004 | Tumake Logot | Zubeen Garg | Bilaval | 0.9349 | 0.0000 |
| BIHU_005 | Kinu Sawonire Sala | Zubeen Garg | Desi | 0.9378 | 0.0024 |
| BIHU_006 | Dikhow Noi Eribo | Zubeen Garg | Desh | 0.9435 | 0.0036 |
| BIHU_007 | Phool Phulise | Zubeen Garg | Desh | 0.9279 | 0.0037 |
| BIHU_008 | Porbotor Dhekiya | Zubeen Garg | Desh | 0.9392 | 0.0013 |
| BIHU_009 | O Nahor | Zubeen Garg | Marva | 0.9373 | 0.0055 |
| BIHU_010 | Janmoni Xopunote Ahiba | Zubeen Garg | Bilaskhani Todi | 0.9465 | 0.0010 |
| BIHU_011 | Eijon Korenu Nasoni | Bhupen Hazarika | Adana | 0.9127 | 0.0014 |
| BIHU_012 | Laje Moru Moru | Bhupen Hazarika | Bhoopali | 0.9350 | 0.0002 |
| BIHU_013 | Ki Sai Aso | Bhupen Hazarika | Ahir Bhairav | 0.9378 | 0.0001 |
| BIHU_014 | Bihureno Birina | Bhupen Hazarika | Marva | 0.9399 | 0.0013 |
| BIHU_015 | Duetar Boraghor | Bhupen Hazarika | Yaman Kalyan | 0.9327 | 0.0013 |
| BIHU_016 | Sangor Chira Pitha | Bhupen Hazarika | Adana | 0.9341 | 0.0004 |
| BIHU_017 | Kopou Phul Phulile | Khagen Mahanta | Bhoopali | 0.9308 | 0.0050 |
| BIHU_018 | Amona Dhanoke | Khagen Mahanta | Desh | 0.9321 | 0.0012 |
| BIHU_019 | O Dhei Oi | Khagen Mahanta | Megh | 0.9416 | 0.0081 |
| BIHU_020 | Dhansiri Pare Hoi | Khagen Mahanta | Bhairavi | 0.9316 | 0.0012 |
| BIHU_021 | Petore Juruni Bhat | Khagen Mahanta | Megh Malhar | 0.9354 | 0.0000 |
| BIHU_022 | Aag Fale Dekhilo | Khagen Mahanta | Bahar | 0.9400 | 0.0025 |
| BIHU_023 | Aghunor Potharot | Khagen Mahanta | Bhoopali | 0.9258 | 0.0022 |
| BIHU_024 | Ebela Hale Bau | Khagen Mahanta | Bhoopali | 0.9140 | 0.0019 |
| BIHU_025 | Senai o Tumake Bisari | Khagen Mahanta | Ahir Bhairav | 0.9319 | 0.0036 |
| BIHU_026 | Bohag Ahil Buli | Shanta Uzir | Ahir Bhairav | 0.9448 | 0.0038 |
| BIHU_027 | Tumi Nasoni | Shanta Uzir | Ahir Bhairav | 0.9331 | 0.0032 |
| BIHU_028 | Xunjone Nuxudhiba | Zubeen Garg, Vitali Das | Bhoopali | 0.9347 | 0.0012 |
| BIHU_029 | Morone Phulonit | Zubeen Garg, Vitali Das | Bhatiyar | 0.9391 | 0.0011 |
| BIHU_030 | Nila Nila Akhore | Zubeen Garg, Nirmali Das | Bilaskhani Todi | 0.9336 | 0.0004 |
| BIHU_031 | Kahili puwate | Zubeen Garg, Priyanka Bharali | Malkauns | 0.9409 | 0.0019 |
| BIHU_032 | Rohare O Rohedoi | Bhitali Das | Sohni | 0.9367 | 0.0018 |
| BIHU_033 | Dhengkoli | Navanita Sharma | Khamaj | 0.9378 | 0.0023 |
| BIHU_034 | Gabhoru Bihu | Deeplina Deka | Khamaj | 0.9449 | 0.0037 |
| BIHU_035 | Bohagi | Rupankrita and Alankrita | Bahar | 0.9335 | 0.0021 |
| BIHU_036 | Podulit Bihu | Rupankrita and Alankrita | Bageshree | 0.9383 | 0.0023 |
| BIHU_037 | Tok Dekhi Mor Gaa | Abhishruti Bezbaruah | Madhuvanti | 0.9381 | 0.0009 |

## Dhamail
- Total recordings: 30
- Successfully classified: 30
- Excluded: 0
- Most frequently retrieved Raag: Darbari
- Second most frequent Raag: Khamaj
- Third most frequent Raag: Adana
- Average Top-1 similarity: 0.9353
- Average Top-1/Top-2 margin: 0.0027

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dithi Das | Darbari | 0.9274 | 0.0019 |
| DHAMAIL_002 | আর যাবো না জলে | Dithi Das | Khamaj | 0.9325 | 0.0012 |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dithi Das | Adana | 0.9343 | 0.0009 |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dithi Das | Lalit | 0.9354 | 0.0008 |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dithi Das | Alhaiya Bilaval | 0.9257 | 0.0014 |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Uma Das | Purba | 0.9376 | 0.0021 |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Uma Das | Abhogi | 0.9279 | 0.0020 |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Uma Das | Bahar | 0.9486 | 0.0010 |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Uma Das | Basant Mukhari | 0.9399 | 0.0007 |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Uma Das | Jaunpuri | 0.9390 | 0.0035 |
| DHAMAIL_011 | বল গো বল সখি | Habib Simu/Shimu | Marva | 0.9381 | 0.0006 |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Habib Simu/Shimu | Darbari | 0.9444 | 0.0038 |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Habib Simu/Shimu | Darbari | 0.9474 | 0.0040 |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Habib Simu/Shimu | Purba | 0.9258 | 0.0025 |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Habib Simu/Shimu | Madhuvanti | 0.9452 | 0.0070 |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Jumiar Shanta | Bihag | 0.9409 | 0.0001 |
| DHAMAIL_017 | বলো গো বলো সই | Pausali Banerjee | Khamaj | 0.9351 | 0.0021 |
| DHAMAIL_018 | কত আদরে আদরে | Pausali Banerjee | Bhupal Todi | 0.9340 | 0.0053 |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Pausali Banerjee | Darbari | 0.9376 | 0.0042 |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Pausali Banerjee | Darbari | 0.9275 | 0.0030 |
| DHAMAIL_021 | জলের ঘাটে | Pausali Banerjee | Bihag | 0.9414 | 0.0115 |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Maadal | Poorvi | 0.9384 | 0.0015 |
| DHAMAIL_023 | Tara Lowo Go Jol | Maadal | Khamaj | 0.9349 | 0.0052 |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Maadal | Desi | 0.9446 | 0.0026 |
| DHAMAIL_025 | Nrittangon Sunamganj | Debdas Chowdhury | Bhairavi | 0.9243 | 0.0025 |
| DHAMAIL_026 | জলার জলার ঘাটে | Himel Dev Rahul | Marva | 0.9157 | 0.0002 |
| DHAMAIL_027 | অচেনা এক রূপবতী | Samim Siddiki | Ahir Bhairav | 0.9444 | 0.0009 |
| DHAMAIL_028 | কুঞ্জের মাঝে | Madol Folk Band | Megh | 0.9370 | 0.0027 |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Suman Dewan | Adana | 0.9412 | 0.0028 |
| DHAMAIL_030 | জানো নি গো সই | Jabel Raza | Asavari | 0.9140 | 0.0021 |

## Garba
- Total recordings: 31
- Successfully classified: 31
- Excluded: 0
- Most frequently retrieved Raag: Adana
- Second most frequent Raag: Megh
- Third most frequent Raag: Alhaiya Bilaval
- Average Top-1 similarity: 0.9394
- Average Top-1/Top-2 margin: 0.0321

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Diwaliben Bhil | Malkauns | 0.9390 | 0.0006 |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Vibha Desai, Sharavan Madhurai | Ahir Bhairav | 0.9412 | 0.0019 |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Geeta Rabari | Alhaiya Bilaval | 0.9447 | 0.0010 |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Lalita Ghodadra | Megh | 0.9466 | 0.0009 |
| GARBA_005 | Amba Aavo To Ramiye | Lalita Ghodadra | Adana | 0.9489 | 0.0047 |
| GARBA_006 | Sonal Garbo Shire | Kishor Manraja | Megh Malhar | 0.9469 | 0.0003 |
| GARBA_007 | Maa No Garbo Re | Geeta Rabari | Asavari | 0.9410 | 0.0009 |
| GARBA_008 | Unchi Talavdi Ni Kor | Asha Bhosle | Asavari | 0.9397 | 0.0053 |
| GARBA_009 | Aavya Maa Na Norta | Pamela Jain, Divya Kumar | Adana | 0.9572 | 0.0027 |
| GARBA_010 | Khodiyar Maa Nu Holdu | Kinjal Dave | Alhaiya Bilaval | 0.9336 | 0.0017 |
| GARBA_011 | Sathiya Puravo | Damyanti Bardai | Megh | 0.9418 | 0.0017 |
| GARBA_012 | Amba Aave Aarasur Vadi | Meena Patel | Yaman or Kalyan | 0.9249 | 0.0055 |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Meena Patel, Praful Dave | Megh | 0.9419 | 0.0008 |
| GARBA_014 | Garba No Rang Lagiyo | Meena Patel, Praful Dave | Miyan Ki Todi orTodi or Darbari Todi | 0.9382 | 0.0003 |
| GARBA_015 | Hindole Zule Amba | Meena Patel | Bhoopali | 0.9291 | 0.0007 |
| GARBA_016 | Jay Ambe Jagdambe Maa | Mital Gadhvi | Adana | 0.9402 | 0.0023 |
| GARBA_017 | Halaji Tara Hath Vakhanu | Aditya Gadhvi | Darbari | 0.9499 | 0.0048 |
| GARBA_018 | Dholida Dhol Re Vagad | Farida Mir | Bhoopali | 0.9344 | 0.0007 |
| GARBA_019 | Pankhida O Pankhida | Hemant Chauhan | Jaunpuri | 0.9259 | 0.0016 |
| GARBA_020 | Tu To Kali Ne Kalyani | Hemant Chauhan | Barwa | 0.9305 | 0.0024 |
| GARBA_021 | He Jagjanani | Hemant Chauhan | Kafi | 0.9288 | 0.0038 |
| GARBA_022 | Chotile Dakla Vagya | Hemant Chauhan | Desh | 0.9191 | 0.0040 |
| GARBA_023 | Jay Aadhyashakti | Hemant Chauhan | Purba | 0.9348 | 0.9348 |
| GARBA_024 | Dam Dam Dakla Vagya | Hemant Chauhan | Bahar | 0.9234 | 0.0006 |
| GARBA_025 | Sonala Vatakadi | Praful Dave, Damyanti Bardai | Marva | 0.9513 | 0.0028 |
| GARBA_026 | Kaho Poonam Na Chand | Shailendra Bharti | Adana | 0.9549 | 0.0003 |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Praful Dave, Meena Patel | Basant | 0.9439 | 0.0018 |
| GARBA_028 | Lejo Rasiya Re Rumal | Arvind Barot, Vatsala Patil | Marva | 0.9475 | 0.0039 |
| GARBA_029 | Pavana Dungre | Praful Dave | Darbari | 0.9384 | 0.0005 |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Arvind Barot | Khamaj | 0.9388 | 0.0008 |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Arvind Barot, Bhavna Rana | Nat Bhairav | 0.9456 | 0.0002 |

## Ghoomar
- Total recordings: 30
- Successfully classified: 30
- Excluded: 0
- Most frequently retrieved Raag: Bhairav
- Second most frequent Raag: Adana
- Third most frequent Raag: Asavari
- Average Top-1 similarity: 0.9361
- Average Top-1/Top-2 margin: 0.0030

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| GHOOMAR_001 | Ghoomar | Seema Mishra | Khamaj | 0.9272 | 0.0009 |
| GHOOMAR_002 | Gangore/Gangaur | Seema Mishra | Bhairav | 0.9523 | 0.0064 |
| GHOOMAR_003 | Choumaso | Seema Mishra | Yaman or Kalyan | 0.9322 | 0.0006 |
| GHOOMAR_004 | More Bole Re | Seema Mishra | Bhairav | 0.9425 | 0.0045 |
| GHOOMAR_005 | Podina | Seema Mishra | Asavari | 0.9418 | 0.0030 |
| GHOOMAR_006 | Talariya Magriya | Seema Mishra | Adana | 0.9249 | 0.0032 |
| GHOOMAR_007 | Panihari Ji | Seema Mishra | Bhoopali | 0.9394 | 0.0046 |
| GHOOMAR_008 | Roomal/Rumal | Seema Mishra | Bilaval | 0.9330 | 0.0004 |
| GHOOMAR_009 | Gorband | Seema Mishra | Bhairav | 0.9341 | 0.0036 |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Seema Mishra | Bhairav | 0.9440 | 0.0039 |
| GHOOMAR_011 | Hivde Ro Haar | Seema Mishra | Barwa | 0.9375 | 0.0021 |
| GHOOMAR_012 | Chirmi | Seema Mishra | Yaman Kalyan | 0.9401 | 0.0022 |
| GHOOMAR_013 | Hichki (Aave Hichki) | Seema Mishra | Darbari | 0.9553 | 0.0054 |
| GHOOMAR_014 | Dal Badal | Seema Mishra | Adana | 0.9427 | 0.0025 |
| GHOOMAR_015 | Samdariyo | Seema Mishra | Miyan Ki Todi orTodi or Darbari Todi | 0.9456 | 0.0002 |
| GHOOMAR_016 | Jeero | Seema Mishra | Asavari | 0.9399 | 0.0003 |
| GHOOMAR_017 | Loonga Ro Batawo | Seema Mishra | Darbari | 0.9259 | 0.0032 |
| GHOOMAR_018 | Taran Ri Chunadi | Seema Mishra | Adana | 0.9214 | 0.0008 |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Seema Mishra | Alhaiya Bilaval | 0.9360 | 0.0156 |
| GHOOMAR_020 | Kangasiyo | Seema Mishra | Ahir Bhairav | 0.9296 | 0.0062 |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Seema Mishra | Bhairav | 0.9455 | 0.0010 |
| GHOOMAR_022 | Kesariya Ro Pag | Seema Mishra | Asavari | 0.9419 | 0.0046 |
| GHOOMAR_023 | Naina Ra Lobhi | Seema Mishra | Miyan Ki Todi orTodi or Darbari Todi | 0.9314 | 0.0009 |
| GHOOMAR_024 | Bajudan Ri Loom | Seema Mishra | Adana | 0.9280 | 0.0010 |
| GHOOMAR_025 | Lehariyo | Seema Mishra | Darbari | 0.9203 | 0.0010 |
| GHOOMAR_026 | Ital Peetal | Seema Mishra | Durga | 0.9281 | 0.0027 |
| GHOOMAR_027 | Moriya | Seema Mishra | Bhatiyar | 0.9387 | 0.0036 |
| GHOOMAR_028 | Cham Cham Chake Chundari | Seema Mishra | Gujari Todi | 0.9231 | 0.0007 |
| GHOOMAR_029 | Hathelyan Re Beech | Seema Mishra | Barwa | 0.9361 | 0.0023 |
| GHOOMAR_030 | Dar Lanjo | Seema Mishra | Bhairav | 0.9436 | 0.0017 |

## Jhumur
- Total recordings: 30
- Successfully classified: 30
- Excluded: 0
- Most frequently retrieved Raag: Bageshree
- Second most frequent Raag: Adana
- Third most frequent Raag: Marva
- Average Top-1 similarity: 0.9371
- Average Top-1/Top-2 margin: 0.0019

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| JHUMUR_001 | Kati Khane Bole | Dulal Manki | Marva | 0.9283 | 0.0022 |
| JHUMUR_0010 | Ekadoshi Ratiya | Zubeen Garg | Lalit | 0.9387 | 0.0008 |
| JHUMUR_0011 | Barire Simare | Zubeen Garg | Ahir Bhairav | 0.9385 | 0.0041 |
| JHUMUR_0012 | Mor Dhoni Go | Zubeen Garg | Malkauns | 0.9293 | 0.0101 |
| JHUMUR_0013 | Jhuli Jhule | Zubeen Garg | Desi | 0.9251 | 0.0000 |
| JHUMUR_0014 | Kansiya | Amrit Tanti | Bageshree | 0.9427 | 0.0009 |
| JHUMUR_0015 | Jhatka | Amrit Tanti | Bhoopali | 0.9381 | 0.0015 |
| JHUMUR_0016 | Gohona | Amrit Tanti | Bageshree | 0.9419 | 0.0032 |
| JHUMUR_0017 | Hariya | Amrit Tanti | Kedar | 0.9342 | 0.0039 |
| JHUMUR_0018 | Koromoti | Amrit Tanti | Khamaj | 0.9395 | 0.0005 |
| JHUMUR_0019 | Sonamoni | Amrit Tanti | Adana | 0.9368 | 0.0002 |
| JHUMUR_002 | Nistei | Dulal Manki | Bageshree | 0.9418 | 0.0000 |
| JHUMUR_0020 | Chaliya Chhodi | Amrit Tanti | Bhoopali | 0.9277 | 0.0030 |
| JHUMUR_0021 | Prem Nadi Tire Tire | Shaan Tanti | Bhairavi | 0.9313 | 0.0002 |
| JHUMUR_0022 | Deewana | Shaan Tanti | Marva | 0.9413 | 0.0035 |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Navin Tanti | Lalit | 0.9528 | 0.0000 |
| JHUMUR_0024 | Laal Pari | Bejnath Tanti | Madhuvanti | 0.9358 | 0.0000 |
| JHUMUR_0025 | Tore Bina | Bejnath Tanti | Asavari | 0.9524 | 0.0003 |
| JHUMUR_0026 | Koromake Akhora | Bejnath Tanti | Bilaval | 0.9437 | 0.0023 |
| JHUMUR_0027 | The Jhumur Song | Neel Akash | Nat Bhairav | 0.9461 | 0.0009 |
| JHUMUR_0028 | Genda Phool | Neel Akash, Jyotika Baruha | Bageshree | 0.9461 | 0.0030 |
| JHUMUR_0029 | Ja Ja Re Mini | Neel Akash | Ahir Bhairav | 0.9344 | 0.0010 |
| JHUMUR_003 | Kuilu Kuilu | Dulal Manki | Darbari | 0.9368 | 0.0021 |
| JHUMUR_0030 | Goya | Neel Akash | Adana | 0.9377 | 0.0008 |
| JHUMUR_004 | Sajanigo | Dulal Manki | Desi | 0.9391 | 0.0063 |
| JHUMUR_005 | Sol Sokhi | Dulal Manki | Deepak | 0.9357 | 0.0023 |
| JHUMUR_006 | Aji Ei Suprabhate | Dulal Manki | Bilaval | 0.9240 | 0.0016 |
| JHUMUR_007 | Kale Gori Muke | Dulal Manki | Nat Bhairav | 0.9320 | 0.0011 |
| JHUMUR_008 | Tohe Gori Nawa | Dulal Manki | Bahar | 0.9298 | 0.0018 |
| JHUMUR_009 | Moi to Jabo Kam Jari | Zubeen Garg | Adana | 0.9310 | 0.0007 |

## Lavani
- Total recordings: 35
- Successfully classified: 35
- Excluded: 0
- Most frequently retrieved Raag: Bahar
- Second most frequent Raag: Marva
- Third most frequent Raag: suddha kalyan
- Average Top-1 similarity: 0.9384
- Average Top-1/Top-2 margin: 0.0021

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Sulochana Chavan | Desh | 0.9381 | 0.0007 |
| LAVANI_002 | Padarawarti Jartaricha | Sulochana Chavan | Marva | 0.9466 | 0.0012 |
| LAVANI_003 | Solava Varis Dhokyacha | Sulochana Chavan | Bahar | 0.9312 | 0.0006 |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Sulochana Chavan | Bahar | 0.9427 | 0.0017 |
| LAVANI_005 | Naav Gaav Kashala Pustay | Sulochana Chavan | Nat Bhairav | 0.9527 | 0.0027 |
| LAVANI_006 | Kalidar Kapuri Paan | Sulochana Chavan | Bahar | 0.9247 | 0.0031 |
| LAVANI_007 | Mala London Foreign Dakhwa | Sulochana Chavan | Malkauns | 0.9369 | 0.0053 |
| LAVANI_008 | Disla Ga Bai Disla | Usha Mangeshkar | Kedar | 0.9498 | 0.0005 |
| LAVANI_009 | Chabidar Chabi | Usha Mangeshkar | suddha kalyan | 0.9305 | 0.0029 |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Usha Mangeshkar | Asavari | 0.9530 | 0.0036 |
| LAVANI_011 | Pikala Jambhul Todu Naka | Usha Mangeshkar | Asavari | 0.9584 | 0.0026 |
| LAVANI_012 | Naki Doli Chhaan | Usha Mangeshkar | Puriya | 0.9578 | 0.0084 |
| LAVANI_013 | Aai Majhya Lagnachi | Usha Mangeshkar | Malhar | 0.9455 | 0.0017 |
| LAVANI_014 | Wajle Ki Bara | Bela Shende | Bahar | 0.9337 | 0.0004 |
| LAVANI_015 | Apsara Aali | Bela Shende | Basant | 0.9305 | 0.0008 |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Bela Shende | Shree | 0.9252 | 0.0007 |
| LAVANI_017 | Kashyala Lavato | Bela Shende | Barwa | 0.9223 | 0.0086 |
| LAVANI_018 | Chhatishi Kavlun Dhara | Surekha Punekar | Bahar | 0.9357 | 0.0007 |
| LAVANI_019 | Jhampar Majha Dheel | Surekha Punekar | Yaman Kalyan | 0.9265 | 0.0015 |
| LAVANI_020 | Themba Thembana Kelai | Surekha Punekar | suddha kalyan | 0.9305 | 0.0029 |
| LAVANI_021 | Sola Hazarat | Uttara Kelkar | Marva | 0.9375 | 0.0025 |
| LAVANI_022 | Hi Natrangi Naar | Ravindra Sathe | Puriya | 0.9424 | 0.0019 |
| LAVANI_023 | Saaman Avjad Majha | Anand Shinde | Yaman Kalyan | 0.9265 | 0.0046 |
| LAVANI_024 | Majha Taat Kar Aadhi | Anand Shinde | suddha kalyan | 0.9290 | 0.0008 |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Asha Bhosle | Yaman or Kalyan | 0.9356 | 0.0006 |
| LAVANI_026 | Bai Mazi Karangali Modali | Asha Bhosle | Basant Mukhari | 0.9551 | 0.0001 |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Pushpa Pagdhare | Marva | 0.9522 | 0.0051 |
| LAVANI_028 | Navi Navi Navari Mee | Pushpa Pagdhare | Malhar | 0.9410 | 0.0045 |
| LAVANI_029 | Ugvali Shukrachi Chandni | Aarti Anklikar | Ahir Bhairav | 0.9206 | 0.0003 |
| LAVANI_030 | Ghotala Ghotala Zala | Vaishali Samant | Bilaskhani Todi | 0.9362 | 0.0001 |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Urmila Dhangkar | Adana | 0.9340 | 0.0014 |
| LAVANI_032 | Aase Wajwa Ki | Bela Shende and chorus | Megh | 0.9426 | 0.0005 |
| LAVANI_033 | Dolyavar Gogal Lava | Various Artists | Darbari | 0.9348 | 0.0002 |
| LAVANI_034 | Reshmachya Reghani | Asha Bhosle | Khamaj | 0.9433 | 0.0004 |
| LAVANI_035 | Chala Jejurila Jau | Uttara Kelkar | Yaman Kalyan | 0.9391 | 0.0006 |

## Maand
- Total recordings: 39
- Successfully classified: 39
- Excluded: 0
- Most frequently retrieved Raag: Darbari
- Second most frequent Raag: Purba
- Third most frequent Raag: Khamaj
- Average Top-1 similarity: 0.9348
- Average Top-1/Top-2 margin: 0.0022

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| MAAND_001 | Kesariya Balam Aavo Ni | Allah Jilai Bai | Darbari | 0.9456 | 0.0034 |
| MAAND_002 | Moray Aaj Aaye Saiyan | Allah Jilai Bai | Hemant | 0.9179 | 0.0005 |
| MAAND_003 | Moomal | Allah Jilai Bai | Madhuvanti | 0.9465 | 0.0012 |
| MAAND_004 | Ghoomar | Allah Jilai Bai | Purba | 0.9283 | 0.0047 |
| MAAND_005 | Bai Sa Ra Beera | Allah Jilai Bai | Megh | 0.9415 | 0.0008 |
| MAAND_006 | Jawai Balha Lagoji | Allah Jilai Bai | Bahar | 0.9220 | 0.0007 |
| MAAND_007 | Kesariya Balam Aavo Ni | Gavri Devi | Darbari | 0.9233 | 0.0002 |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Seema Mishra | Bhatiyar | 0.9289 | 0.0002 |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Seema Mishra | Khamaj | 0.9444 | 0.0013 |
| MAAND_010 | Badila Yahi Rahiyo Saa | Banarsi Babu | Gauri | 0.9537 | 0.0080 |
| MAAND_011 | Taniya Resham Ki | Banarsi Babu | Adana | 0.9454 | 0.0024 |
| MAAND_012 | Kad Janulee Sasre | Rehana Mirza | Alhaiya Bilaval | 0.9421 | 0.0034 |
| MAAND_013 | Le Dekh Nangi Melo | Rehana Mirza | Khamaj | 0.9477 | 0.0011 |
| MAAND_014 | The To Piyo Sadarudi | Rehana Mirza | Rageshree | 0.9392 | 0.0007 |
| MAAND_015 | Kesariya Balam (Desert Slide) | Anwar Khan | Miyan Ki Todi orTodi or Darbari Todi | 0.9267 | 0.0029 |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Anwar Khan | Bhairavi | 0.9186 | 0.0039 |
| MAAND_017 | Apne Desh Ri Aajadi | Anwar Khan | Sorath | 0.9351 | 0.0038 |
| MAAND_018 | Kesariya Balam Aao Ni | Anwar Khan | Ahir Bhairav | 0.9354 | 0.0035 |
| MAAND_019 | Raag Maand Bandish | Mame Khan | Bilaval | 0.9388 | 0.0000 |
| MAAND_020 | Rajasthani Folk Raag Maand | Mame Khan | Bilaval | 0.9409 | 0.0012 |
| MAAND_021 | Chaudhary | Mame Khan | Ahir Bhairav | 0.9324 | 0.0017 |
| MAAND_022 | Chhap Tilak Sab Cheeni | Kirti Sagathia | Adana | 0.9323 | 0.0001 |
| MAAND_023 | Balam | Swaroop Khan | Darbari | 0.9495 | 0.0044 |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Swaroop Khan Sura | Marva | 0.9147 | 0.0011 |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Sikander Khan Baru | Khamaj | 0.9313 | 0.0017 |
| MAAND_026 | Darude Ro Rog Lagiyo | Sikander Khan Baru | Ahir Bhairav | 0.9131 | 0.0033 |
| MAAND_027 | Kesariya Balam Padharo | Bhungar Khan | Komal Bhairav | 0.9439 | 0.0038 |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Kheta Khan, Fakira Khan Bhadresh | Kafi | 0.9416 | 0.0019 |
| MAAND_029 | Kalali | Latif Khan | Malkauns | 0.9186 | 0.0017 |
| MAAND_030 | Kesariya Balam | Raitila (Group) | Yaman Kalyan | 0.9278 | 0.0014 |
| MAAND_031 | Joori Thaane Na Maare Raad | Jameela Bai & Kulsum Bai | Yaman Kalyan | 0.9218 | 0.0007 |
| MAAND_032 | Alija Dhola Ekladi | Jameela Bai & Kulsum Bai | Desh | 0.9228 | 0.0037 |
| MAAND_033 | Mumal | Jameela Bai & Kulsum Bai | Purba | 0.9342 | 0.0012 |
| MAAND_034 | Mhari Sajni Ae | Nand Lal Bhat | Sorath | 0.9439 | 0.0019 |
| MAAND_035 | Rajwadi Mehfil | Nand Lal Bhat | Alhaiya Bilaval | 0.9365 | 0.0002 |
| MAAND_036 | Thane Samjhaye Sakha Sa | Munshi Khan | Darbari | 0.9420 | 0.0025 |
| MAAND_037 | Padharo Mhare Mehal | Munshi Khan | Bhupal Todi | 0.9445 | 0.0027 |
| MAAND_038 | Daru Dakha Ro | Munshi Khan | Gauri | 0.9424 | 0.0025 |
| MAAND_039 | Mahara Sajan Base | Munshi Khan | Purba | 0.9428 | 0.0053 |

## Sambalpuri
- Total recordings: 34
- Successfully classified: 33
- Excluded: 1
- Most frequently retrieved Raag: Bilaval
- Second most frequent Raag: Sorath
- Third most frequent Raag: Darbari
- Average Top-1 similarity: 0.9379
- Average Top-1/Top-2 margin: 0.0022

| Song ID | Song | Artist | Predicted Raag | Similarity | Margin |
|---|---|---|---|---:|---:|
| SAMBALPURI_001 | Nabu Ki Sambalpur | Human Sagar | Ahir Bhairav | 0.9407 | 0.0044 |
| SAMBALPURI_002 | Tor Lagi Marijibi | Human Sagar | Bilaval | 0.9375 | 0.0008 |
| SAMBALPURI_003 | Sacha Lover | Human Sagar | Basant | 0.9391 | 0.0021 |
| SAMBALPURI_004 | Pardesia Raja | Human Sagar | Sorath | 0.9508 | 0.0017 |
| SAMBALPURI_005 | Lal Sadhi | Human Sagar | Bilaval | 0.9279 | 0.0025 |
| SAMBALPURI_006 | Rasia | Mantu Chhuria | Bilaval | 0.9351 | 0.0074 |
| SAMBALPURI_007 | A Mo Dil | Mantu Chhuria | Darbari | 0.9364 | 0.0034 |
| SAMBALPURI_008 | Hae Go | Mantu Chhuria | Darbari | 0.9347 | 0.0002 |
| SAMBALPURI_009 | Aukaat | Mantu Chhuria | Megh | 0.9504 | 0.0008 |
| SAMBALPURI_010 | Rani Guri | Mantu Chhuria | Bageshree | 0.9397 | 0.0030 |
| SAMBALPURI_011 | Gidja Gija Re | Asima Panda | Khamaj | 0.9379 | 0.0003 |
| SAMBALPURI_012 | Tui Bewafa | Asima Panda | Desh | 0.9439 | 0.0037 |
| SAMBALPURI_013 | Ramagaru | Asima Panda | Sorath | 0.9276 | 0.0001 |
| SAMBALPURI_014 | Deewana Banalu Rani | Asima Panda | Bageshree | 0.9338 | 0.0002 |
| SAMBALPURI_015 | SambalPuri Saiyaan | Asima Panda | Sorath | 0.9390 | 0.0016 |
| SAMBALPURI_016 | Anta Halei Halei | Ruku Suna | Alhaiya Bilaval | 0.9476 | 0.0009 |
| SAMBALPURI_017 | Chandir Paenjhal | Ruku Suna | Adana | 0.9446 | 0.0003 |
| SAMBALPURI_018 | Prem Paga | Ruku Suna | Darbari | 0.9492 | 0.0023 |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Ruku Suna | Megh | 0.9427 | 0.0007 |
| SAMBALPURI_020 | A Labanga Lata | Ruku Suna | Bhairavi | 0.9297 | 0.0006 |
| SAMBALPURI_021 | Badnam | Prakash Jal | Rageshree | 0.9225 | 0.0040 |
| SAMBALPURI_022 | Dil Bhitre Lekhideli | Prakash Jal | Pahadi | 0.9342 | 0.0023 |
| SAMBALPURI_023 | Bewafa Mita | Prakash Jal | Adana | 0.9353 | 0.0013 |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Prakash Jal | Pahadi | 0.9374 | 0.0006 |
| SAMBALPURI_025 | Le Le Baula | Antara Chakrabarty | Asavari | 0.9484 | 0.0017 |
| SAMBALPURI_026 | Chahanewali | Antara Chakrabarty | NO PREDICTION | N/A | N/A |
| SAMBALPURI_027 | Feel My Love | Antara Chakrabarty | Marva | 0.9430 | 0.0005 |
| SAMBALPURI_028 | Kulfirani Chocobar | Ira Mohanty | Marva | 0.9496 | 0.0067 |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Ira Mohanty | Khamaj | 0.9419 | 0.0053 |
| SAMBALPURI_030 | Sundri Janha | Ira Mohanty | Bageshree | 0.9371 | 0.0015 |
| SAMBALPURI_031 | Rangabati | Jitendra Haripal | Bahar | 0.9151 | 0.0038 |
| SAMBALPURI_032 | Aha Mor Galar Malire | Jitendra Haripal | Basant | 0.9320 | 0.0052 |
| SAMBALPURI_033 | Din Gani Gani | Jitendra Haripal | Shree | 0.9340 | 0.0001 |
| SAMBALPURI_034 | Namo Namo Ago | Jitendra Haripal | Bhairav | 0.9314 | 0.0009 |

## PART 5 — RAAG-CENTRIC INDEX

## Raag → Folk Recordings

### Abhogi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul | 0.9225 | 0.0025 |
| BAUL_015 | Aami Kothay Pabo Tare | Baul | 0.9181 | 0.0009 |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul | 0.9294 | 0.0005 |
| BHANGRA_029 | Putt Jattan De | Bhangra | 0.9170 | 0.0002 |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail | 0.9279 | 0.0020 |

### Adana
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_006 | Munda Ki Mangda | Bhangra | 0.9311 | 0.0029 |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra | 0.9344 | 0.0046 |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra | 0.9397 | 0.0015 |
| BIHU_011 | Eijon Korenu Nasoni | Bihu | 0.9127 | 0.0014 |
| BIHU_016 | Sangor Chira Pitha | Bihu | 0.9341 | 0.0004 |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail | 0.9343 | 0.0009 |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Dhamail | 0.9412 | 0.0028 |
| GARBA_005 | Amba Aavo To Ramiye | Garba | 0.9489 | 0.0047 |
| GARBA_009 | Aavya Maa Na Norta | Garba | 0.9572 | 0.0027 |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba | 0.9402 | 0.0023 |
| GARBA_026 | Kaho Poonam Na Chand | Garba | 0.9549 | 0.0003 |
| GHOOMAR_006 | Talariya Magriya | Ghoomar | 0.9249 | 0.0032 |
| GHOOMAR_014 | Dal Badal | Ghoomar | 0.9427 | 0.0025 |
| GHOOMAR_018 | Taran Ri Chunadi | Ghoomar | 0.9214 | 0.0008 |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar | 0.9280 | 0.0010 |
| JHUMUR_0019 | Sonamoni | Jhumur | 0.9368 | 0.0002 |
| JHUMUR_0030 | Goya | Jhumur | 0.9377 | 0.0008 |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur | 0.9310 | 0.0007 |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Lavani | 0.9340 | 0.0014 |
| MAAND_011 | Taniya Resham Ki | Maand | 0.9454 | 0.0024 |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand | 0.9323 | 0.0001 |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri | 0.9446 | 0.0003 |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri | 0.9353 | 0.0013 |

### Ahir Bhairav
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_013 | Ki Sai Aso | Bihu | 0.9378 | 0.0001 |
| BIHU_025 | Senai o Tumake Bisari | Bihu | 0.9319 | 0.0036 |
| BIHU_026 | Bohag Ahil Buli | Bihu | 0.9448 | 0.0038 |
| BIHU_027 | Tumi Nasoni | Bihu | 0.9331 | 0.0032 |
| DHAMAIL_027 | অচেনা এক রূপবতী | Dhamail | 0.9444 | 0.0009 |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba | 0.9412 | 0.0019 |
| GHOOMAR_020 | Kangasiyo | Ghoomar | 0.9296 | 0.0062 |
| JHUMUR_0011 | Barire Simare | Jhumur | 0.9385 | 0.0041 |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur | 0.9344 | 0.0010 |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani | 0.9206 | 0.0003 |
| MAAND_018 | Kesariya Balam Aao Ni | Maand | 0.9354 | 0.0035 |
| MAAND_021 | Chaudhary | Maand | 0.9324 | 0.0017 |
| MAAND_026 | Darude Ro Rog Lagiyo | Maand | 0.9131 | 0.0033 |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri | 0.9407 | 0.0044 |

### Alhaiya Bilaval
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra | 0.9291 | 0.0005 |
| BHANGRA_023 | Bhaina Nachiyaan | Bhangra | 0.9301 | 0.0049 |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail | 0.9257 | 0.0014 |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Garba | 0.9447 | 0.0010 |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba | 0.9336 | 0.0017 |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar | 0.9360 | 0.0156 |
| MAAND_012 | Kad Janulee Sasre | Maand | 0.9421 | 0.0034 |
| MAAND_035 | Rajwadi Mehfil | Maand | 0.9365 | 0.0002 |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri | 0.9476 | 0.0009 |

### Asavari
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_002 | Toi Nagini Ne | Bihu | 0.9280 | 0.0009 |
| DHAMAIL_030 | জানো নি গো সই | Dhamail | 0.9140 | 0.0021 |
| GARBA_007 | Maa No Garbo Re | Garba | 0.9410 | 0.0009 |
| GARBA_008 | Unchi Talavdi Ni Kor | Garba | 0.9397 | 0.0053 |
| GHOOMAR_005 | Podina | Ghoomar | 0.9418 | 0.0030 |
| GHOOMAR_016 | Jeero | Ghoomar | 0.9399 | 0.0003 |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar | 0.9419 | 0.0046 |
| JHUMUR_0025 | Tore Bina | Jhumur | 0.9524 | 0.0003 |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani | 0.9530 | 0.0036 |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani | 0.9584 | 0.0026 |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri | 0.9484 | 0.0017 |

### Bageshree
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_036 | Podulit Bihu | Bihu | 0.9383 | 0.0023 |
| JHUMUR_0014 | Kansiya | Jhumur | 0.9427 | 0.0009 |
| JHUMUR_0016 | Gohona | Jhumur | 0.9419 | 0.0032 |
| JHUMUR_002 | Nistei | Jhumur | 0.9418 | 0.0000 |
| JHUMUR_0028 | Genda Phool | Jhumur | 0.9461 | 0.0030 |
| SAMBALPURI_010 | Rani Guri | Sambalpuri | 0.9397 | 0.0030 |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri | 0.9338 | 0.0002 |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri | 0.9371 | 0.0015 |

### Bahar
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul | 0.9370 | 0.0031 |
| BHANGRA_001 | Tera Yaar Bolda | Bhangra | 0.9212 | 0.0001 |
| BIHU_022 | Aag Fale Dekhilo | Bihu | 0.9400 | 0.0025 |
| BIHU_035 | Bohagi | Bihu | 0.9335 | 0.0021 |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail | 0.9486 | 0.0010 |
| GARBA_024 | Dam Dam Dakla Vagya | Garba | 0.9234 | 0.0006 |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur | 0.9298 | 0.0018 |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani | 0.9312 | 0.0006 |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani | 0.9427 | 0.0017 |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani | 0.9247 | 0.0031 |
| LAVANI_014 | Wajle Ki Bara | Lavani | 0.9337 | 0.0004 |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani | 0.9357 | 0.0007 |
| MAAND_006 | Jawai Balha Lagoji | Maand | 0.9220 | 0.0007 |
| SAMBALPURI_031 | Rangabati | Sambalpuri | 0.9151 | 0.0038 |

### Barwa
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_009 | Kon Ba Zharir | Baul | 0.9362 | 0.0012 |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba | 0.9305 | 0.0024 |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar | 0.9375 | 0.0021 |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar | 0.9361 | 0.0023 |
| LAVANI_017 | Kashyala Lavato | Lavani | 0.9223 | 0.0086 |

### Basant
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba | 0.9439 | 0.0018 |
| LAVANI_015 | Apsara Aali | Lavani | 0.9305 | 0.0008 |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri | 0.9391 | 0.0021 |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri | 0.9320 | 0.0052 |

### Basant Mukhari
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_032 | Jeona Morh | Bhangra | 0.9448 | 0.0007 |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail | 0.9399 | 0.0007 |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani | 0.9551 | 0.0001 |

### Bhairav
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_020 | Jemon Beni Temni Rabey | Baul | 0.9400 | 0.0043 |
| GHOOMAR_002 | Gangore/Gangaur | Ghoomar | 0.9523 | 0.0064 |
| GHOOMAR_004 | More Bole Re | Ghoomar | 0.9425 | 0.0045 |
| GHOOMAR_009 | Gorband | Ghoomar | 0.9341 | 0.0036 |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar | 0.9440 | 0.0039 |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar | 0.9455 | 0.0010 |
| GHOOMAR_030 | Dar Lanjo | Ghoomar | 0.9436 | 0.0017 |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri | 0.9314 | 0.0009 |

### Bhairavi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_005 | Amar Aei Rudir Dhara | Baul | 0.9324 | 0.0031 |
| BAUL_011 | Chor Poreche Babur Bagane | Baul | 0.9281 | 0.0076 |
| BAUL_014 | Dekhechhi Rup-Sagare | Baul | 0.9367 | 0.0049 |
| BAUL_019 | Aami Ek Pagal Baul | Baul | 0.9208 | 0.0001 |
| BHANGRA_013 | Pao Bhangra | Bhangra | 0.9151 | 0.0011 |
| BIHU_020 | Dhansiri Pare Hoi | Bihu | 0.9316 | 0.0012 |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail | 0.9243 | 0.0025 |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur | 0.9313 | 0.0002 |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand | 0.9186 | 0.0039 |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri | 0.9297 | 0.0006 |

### Bhatiyar
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_029 | Morone Phulonit | Bihu | 0.9391 | 0.0011 |
| GHOOMAR_027 | Moriya | Ghoomar | 0.9387 | 0.0036 |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand | 0.9289 | 0.0002 |

### Bhimpalasi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_017 | Milon Hobe Koto Dine | Baul | 0.9335 | 0.0017 |

### Bhoopali
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_002 | Jatt Di Pasand | Bhangra | 0.9307 | 0.0035 |
| BIHU_012 | Laje Moru Moru | Bihu | 0.9350 | 0.0002 |
| BIHU_017 | Kopou Phul Phulile | Bihu | 0.9308 | 0.0050 |
| BIHU_023 | Aghunor Potharot | Bihu | 0.9258 | 0.0022 |
| BIHU_024 | Ebela Hale Bau | Bihu | 0.9140 | 0.0019 |
| BIHU_028 | Xunjone Nuxudhiba | Bihu | 0.9347 | 0.0012 |
| GARBA_015 | Hindole Zule Amba | Garba | 0.9291 | 0.0007 |
| GARBA_018 | Dholida Dhol Re Vagad | Garba | 0.9344 | 0.0007 |
| GHOOMAR_007 | Panihari Ji | Ghoomar | 0.9394 | 0.0046 |
| JHUMUR_0015 | Jhatka | Jhumur | 0.9381 | 0.0015 |
| JHUMUR_0020 | Chaliya Chhodi | Jhumur | 0.9277 | 0.0030 |

### Bhupal Todi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| DHAMAIL_018 | কত আদরে আদরে | Dhamail | 0.9340 | 0.0053 |
| MAAND_037 | Padharo Mhare Mehal | Maand | 0.9445 | 0.0027 |

### Bihag
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail | 0.9409 | 0.0001 |
| DHAMAIL_021 | জলের ঘাটে | Dhamail | 0.9414 | 0.0115 |

### Bilaskhani Todi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu | 0.9465 | 0.0010 |
| BIHU_030 | Nila Nila Akhore | Bihu | 0.9336 | 0.0004 |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani | 0.9362 | 0.0001 |

### Bilaval
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra | 0.9433 | 0.0009 |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra | 0.9226 | 0.0081 |
| BIHU_004 | Tumake Logot | Bihu | 0.9349 | 0.0000 |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar | 0.9330 | 0.0004 |
| JHUMUR_0026 | Koromake Akhora | Jhumur | 0.9437 | 0.0023 |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur | 0.9240 | 0.0016 |
| MAAND_019 | Raag Maand Bandish | Maand | 0.9388 | 0.0000 |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand | 0.9409 | 0.0012 |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri | 0.9375 | 0.0008 |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri | 0.9279 | 0.0025 |
| SAMBALPURI_006 | Rasia | Sambalpuri | 0.9351 | 0.0074 |

### Darbari
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_003 | Dupatta Tera Satrang Da | Bhangra | 0.9340 | 0.0000 |
| BHANGRA_004 | Wang Wargi Kudi | Bhangra | 0.9286 | 0.0043 |
| BHANGRA_009 | Ho Jayegi Balle Balle | Bhangra | 0.9468 | 0.0071 |
| BHANGRA_010 | Dardi Rab Rab Kardi | Bhangra | 0.9320 | 0.0045 |
| BHANGRA_016 | Mitran Ne Mundri Te | Bhangra | 0.9400 | 0.0006 |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra | 0.9297 | 0.0002 |
| BHANGRA_021 | Kinni Sohni | Bhangra | 0.9421 | 0.0063 |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Bhangra | 0.9309 | 0.0065 |
| BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Bhangra | 0.9342 | 0.0030 |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dhamail | 0.9274 | 0.0019 |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail | 0.9444 | 0.0038 |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Dhamail | 0.9474 | 0.0040 |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Dhamail | 0.9376 | 0.0042 |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail | 0.9275 | 0.0030 |
| GARBA_017 | Halaji Tara Hath Vakhanu | Garba | 0.9499 | 0.0048 |
| GARBA_029 | Pavana Dungre | Garba | 0.9384 | 0.0005 |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar | 0.9553 | 0.0054 |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar | 0.9259 | 0.0032 |
| GHOOMAR_025 | Lehariyo | Ghoomar | 0.9203 | 0.0010 |
| JHUMUR_003 | Kuilu Kuilu | Jhumur | 0.9368 | 0.0021 |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani | 0.9348 | 0.0002 |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand | 0.9456 | 0.0034 |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand | 0.9233 | 0.0002 |
| MAAND_023 | Balam | Maand | 0.9495 | 0.0044 |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand | 0.9420 | 0.0025 |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri | 0.9364 | 0.0034 |
| SAMBALPURI_008 | Hae Go | Sambalpuri | 0.9347 | 0.0002 |
| SAMBALPURI_018 | Prem Paga | Sambalpuri | 0.9492 | 0.0023 |

### Deepak
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| JHUMUR_005 | Sol Sokhi | Jhumur | 0.9357 | 0.0023 |

### Desh
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_008 | Bolo Ta Ra Ra | Bhangra | 0.9417 | 0.0032 |
| BIHU_006 | Dikhow Noi Eribo | Bihu | 0.9435 | 0.0036 |
| BIHU_007 | Phool Phulise | Bihu | 0.9279 | 0.0037 |
| BIHU_008 | Porbotor Dhekiya | Bihu | 0.9392 | 0.0013 |
| BIHU_018 | Amona Dhanoke | Bihu | 0.9321 | 0.0012 |
| GARBA_022 | Chotile Dakla Vagya | Garba | 0.9191 | 0.0040 |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani | 0.9381 | 0.0007 |
| MAAND_032 | Alija Dhola Ekladi | Maand | 0.9228 | 0.0037 |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri | 0.9439 | 0.0037 |

### Desi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_005 | Kinu Sawonire Sala | Bihu | 0.9378 | 0.0024 |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail | 0.9446 | 0.0026 |
| JHUMUR_0013 | Jhuli Jhule | Jhumur | 0.9251 | 0.0000 |
| JHUMUR_004 | Sajanigo | Jhumur | 0.9391 | 0.0063 |

### Durga
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_022 | Emon Mon Byabsaa | Baul | 0.9488 | 0.0040 |
| GHOOMAR_026 | Ital Peetal | Ghoomar | 0.9281 | 0.0027 |

### Gauri
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand | 0.9537 | 0.0080 |
| MAAND_038 | Daru Dakha Ro | Maand | 0.9424 | 0.0025 |

### Gujari Todi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra | 0.9384 | 0.0081 |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra | 0.9387 | 0.0054 |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar | 0.9231 | 0.0007 |

### Hemant
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand | 0.9179 | 0.0005 |

### Hindol
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_003 | Kokalote Tongali | Bihu | 0.9402 | 0.0002 |

### Jaunpuri
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail | 0.9390 | 0.0035 |
| GARBA_019 | Pankhida O Pankhida | Garba | 0.9259 | 0.0016 |

### Kafi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| GARBA_021 | He Jagjanani | Garba | 0.9288 | 0.0038 |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand | 0.9416 | 0.0019 |

### Kedar
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| JHUMUR_0017 | Hariya | Jhumur | 0.9342 | 0.0039 |
| LAVANI_008 | Disla Ga Bai Disla | Lavani | 0.9498 | 0.0005 |

### Khamaj
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_012 | Golemale Gole Male | Baul | 0.9357 | 0.0016 |
| BIHU_033 | Dhengkoli | Bihu | 0.9378 | 0.0023 |
| BIHU_034 | Gabhoru Bihu | Bihu | 0.9449 | 0.0037 |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail | 0.9325 | 0.0012 |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail | 0.9351 | 0.0021 |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail | 0.9349 | 0.0052 |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba | 0.9388 | 0.0008 |
| GHOOMAR_001 | Ghoomar | Ghoomar | 0.9272 | 0.0009 |
| JHUMUR_0018 | Koromoti | Jhumur | 0.9395 | 0.0005 |
| LAVANI_034 | Reshmachya Reghani | Lavani | 0.9433 | 0.0004 |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand | 0.9444 | 0.0013 |
| MAAND_013 | Le Dekh Nangi Melo | Maand | 0.9477 | 0.0011 |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand | 0.9313 | 0.0017 |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri | 0.9379 | 0.0003 |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri | 0.9419 | 0.0053 |

### Komal Bhairav
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| MAAND_027 | Kesariya Balam Padharo | Maand | 0.9439 | 0.0038 |

### Lalit
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail | 0.9354 | 0.0008 |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur | 0.9387 | 0.0008 |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Jhumur | 0.9528 | 0.0000 |

### Madhuvanti
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_001 | Agun Pani | Baul | 0.9390 | 0.0008 |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu | 0.9381 | 0.0009 |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail | 0.9452 | 0.0070 |
| JHUMUR_0024 | Laal Pari | Jhumur | 0.9358 | 0.0000 |
| MAAND_003 | Moomal | Maand | 0.9465 | 0.0012 |

### Malhar
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| LAVANI_013 | Aai Majhya Lagnachi | Lavani | 0.9455 | 0.0017 |
| LAVANI_028 | Navi Navi Navari Mee | Lavani | 0.9410 | 0.0045 |

### Malkauns
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_008 | Tripti | Baul | 0.8812 | 0.0002 |
| BAUL_026 | Sokhi Jamuna Jol | Baul | 0.9320 | 0.0005 |
| BAUL_027 | Pre Loye Jao Amay | Baul | 0.9434 | 0.0032 |
| BHANGRA_025 | Soho Road Utey | Bhangra | 0.9347 | 0.0026 |
| BIHU_031 | Kahili puwate | Bihu | 0.9409 | 0.0019 |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba | 0.9390 | 0.0006 |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur | 0.9293 | 0.0101 |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani | 0.9369 | 0.0053 |
| MAAND_029 | Kalali | Maand | 0.9186 | 0.0017 |

### Marva
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_004 | Ananda Lahari | Baul | 0.9303 | 0.0039 |
| BIHU_009 | O Nahor | Bihu | 0.9373 | 0.0055 |
| BIHU_014 | Bihureno Birina | Bihu | 0.9399 | 0.0013 |
| DHAMAIL_011 | বল গো বল সখি | Dhamail | 0.9381 | 0.0006 |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail | 0.9157 | 0.0002 |
| GARBA_025 | Sonala Vatakadi | Garba | 0.9513 | 0.0028 |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba | 0.9475 | 0.0039 |
| JHUMUR_001 | Kati Khane Bole | Jhumur | 0.9283 | 0.0022 |
| JHUMUR_0022 | Deewana | Jhumur | 0.9413 | 0.0035 |
| LAVANI_002 | Padarawarti Jartaricha | Lavani | 0.9466 | 0.0012 |
| LAVANI_021 | Sola Hazarat | Lavani | 0.9375 | 0.0025 |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani | 0.9522 | 0.0051 |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand | 0.9147 | 0.0011 |
| SAMBALPURI_027 | Feel My Love | Sambalpuri | 0.9430 | 0.0005 |
| SAMBALPURI_028 | Kulfirani Chocobar | Sambalpuri | 0.9496 | 0.0067 |

### Megh
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_007 | Tunak Tunak Tun | Bhangra | 0.9487 | 0.0127 |
| BIHU_019 | O Dhei Oi | Bihu | 0.9416 | 0.0081 |
| DHAMAIL_028 | কুঞ্জের মাঝে | Dhamail | 0.9370 | 0.0027 |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba | 0.9466 | 0.0009 |
| GARBA_011 | Sathiya Puravo | Garba | 0.9418 | 0.0017 |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba | 0.9419 | 0.0008 |
| LAVANI_032 | Aase Wajwa Ki | Lavani | 0.9426 | 0.0005 |
| MAAND_005 | Bai Sa Ra Beera | Maand | 0.9415 | 0.0008 |
| SAMBALPURI_009 | Aukaat | Sambalpuri | 0.9504 | 0.0008 |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri | 0.9427 | 0.0007 |

### Megh Malhar
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_021 | Petore Juruni Bhat | Bihu | 0.9354 | 0.0000 |
| GARBA_006 | Sonal Garbo Shire | Garba | 0.9469 | 0.0003 |

### Miyan Ki Todi orTodi or Darbari Todi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_029 | Aar Amare Marishne Maa | Baul | 0.9414 | 0.0020 |
| GARBA_014 | Garba No Rang Lagiyo | Garba | 0.9382 | 0.0003 |
| GHOOMAR_015 | Samdariyo | Ghoomar | 0.9456 | 0.0002 |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar | 0.9314 | 0.0009 |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand | 0.9267 | 0.0029 |

### Nat Bhairav
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_028 | Somoy Gele Sadhon Hobena | Baul | 0.9180 | 0.0009 |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba | 0.9456 | 0.0002 |
| JHUMUR_0027 | The Jhumur Song | Jhumur | 0.9461 | 0.0009 |
| JHUMUR_007 | Kale Gori Muke | Jhumur | 0.9320 | 0.0011 |
| LAVANI_005 | Naav Gaav Kashala Pustay | Lavani | 0.9527 | 0.0027 |

### Pahadi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| SAMBALPURI_022 | Dil Bhitre Lekhideli | Sambalpuri | 0.9342 | 0.0023 |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri | 0.9374 | 0.0006 |

### Poorvi
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail | 0.9384 | 0.0015 |

### Purba
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_021 | Prem Koraa Ki Jwalaa | Baul | 0.9417 | 0.0084 |
| BAUL_023 | Manush Bhajo Manusheyi Pujo | Baul | 0.9455 | 0.9455 |
| BAUL_024 | Dil Dariyar Majhete | Baul | 0.9334 | 0.0088 |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Baul | 0.9353 | 0.0059 |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail | 0.9376 | 0.0021 |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail | 0.9258 | 0.0025 |
| GARBA_023 | Jay Aadhyashakti | Garba | 0.9348 | 0.9348 |
| MAAND_004 | Ghoomar | Maand | 0.9283 | 0.0047 |
| MAAND_033 | Mumal | Maand | 0.9342 | 0.0012 |
| MAAND_039 | Mahara Sajan Base | Maand | 0.9428 | 0.0053 |

### Puriya
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Baul | 0.9328 | 0.0052 |
| LAVANI_012 | Naki Doli Chhaan | Lavani | 0.9578 | 0.0084 |
| LAVANI_022 | Hi Natrangi Naar | Lavani | 0.9424 | 0.0019 |

### Rageshree
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_019 | Kurri Garam Jayee | Bhangra | 0.9322 | 0.0003 |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra | 0.9379 | 0.0090 |
| MAAND_014 | The To Piyo Sadarudi | Maand | 0.9392 | 0.0007 |
| SAMBALPURI_021 | Badnam | Sambalpuri | 0.9225 | 0.0040 |

### Shree
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_020 | Chal Hun | Bhangra | 0.9341 | 0.0014 |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani | 0.9252 | 0.0007 |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri | 0.9340 | 0.0001 |

### Sohni
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BIHU_001 | Morom Logai Logai | Bihu | 0.9355 | 0.0021 |
| BIHU_032 | Rohare O Rohedoi | Bihu | 0.9367 | 0.0018 |

### Sorath
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_016 | Shesher Deene Sejon Bine | Baul | 0.9355 | 0.0051 |
| MAAND_017 | Apne Desh Ri Aajadi | Maand | 0.9351 | 0.0038 |
| MAAND_034 | Mhari Sajni Ae | Maand | 0.9439 | 0.0019 |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri | 0.9508 | 0.0017 |
| SAMBALPURI_013 | Ramagaru | Sambalpuri | 0.9276 | 0.0001 |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri | 0.9390 | 0.0016 |

### Yaman Kalyan
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BAUL_003 | Kalkatou Mon Heshekeley | Baul | 0.9422 | 0.0029 |
| BAUL_006 | Hari Balo Re | Baul | 0.9325 | 0.0033 |
| BAUL_007 | Gour Lila Bazare | Baul | 0.9373 | 0.0059 |
| BAUL_010 | Sonogo Aian Dada | Baul | 0.9457 | 0.0040 |
| BHANGRA_028 | Jugni | Bhangra | 0.9351 | 0.0043 |
| BIHU_015 | Duetar Boraghor | Bihu | 0.9327 | 0.0013 |
| GHOOMAR_012 | Chirmi | Ghoomar | 0.9401 | 0.0022 |
| LAVANI_019 | Jhampar Majha Dheel | Lavani | 0.9265 | 0.0015 |
| LAVANI_023 | Saaman Avjad Majha | Lavani | 0.9265 | 0.0046 |
| LAVANI_035 | Chala Jejurila Jau | Lavani | 0.9391 | 0.0006 |
| MAAND_030 | Kesariya Balam | Maand | 0.9278 | 0.0014 |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand | 0.9218 | 0.0007 |

### Yaman or Kalyan
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra | 0.9279 | 0.0041 |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba | 0.9249 | 0.0055 |
| GHOOMAR_003 | Choumaso | Ghoomar | 0.9322 | 0.0006 |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani | 0.9356 | 0.0006 |

### suddha kalyan
| Song ID | Song | Genre | Similarity | Margin |
|---|---|---|---:|---:|
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Bhangra | 0.9264 | 0.0035 |
| LAVANI_009 | Chabidar Chabi | Lavani | 0.9305 | 0.0029 |
| LAVANI_020 | Themba Thembana Kelai | Lavani | 0.9305 | 0.0029 |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani | 0.9290 | 0.0008 |

## PART 6 — TOP-5 RETRIEVAL INDEX

### Abhogi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul |
| BAUL_003 | Kalkatou Mon Heshekeley | Baul |
| BAUL_011 | Chor Poreche Babur Bagane | Baul |
| BAUL_012 | Golemale Gole Male | Baul |
| BAUL_015 | Aami Kothay Pabo Tare | Baul |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul |
| BAUL_027 | Pre Loye Jao Amay | Baul |
| BHANGRA_004 | Wang Wargi Kudi | Bhangra |
| BHANGRA_029 | Putt Jattan De | Bhangra |
| BIHU_004 | Tumake Logot | Bihu |
| BIHU_024 | Ebela Hale Bau | Bihu |
| BIHU_027 | Tumi Nasoni | Bihu |
| BIHU_032 | Rohare O Rohedoi | Bihu |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail |
| MAAND_014 | The To Piyo Sadarudi | Maand |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri |

### Adana
| Song ID | Song | Genre |
|---|---|---|
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul |
| BHANGRA_006 | Munda Ki Mangda | Bhangra |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra |
| BHANGRA_021 | Kinni Sohni | Bhangra |
| BHANGRA_028 | Jugni | Bhangra |
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Bhangra |
| BIHU_011 | Eijon Korenu Nasoni | Bihu |
| BIHU_012 | Laje Moru Moru | Bihu |
| BIHU_013 | Ki Sai Aso | Bihu |
| BIHU_016 | Sangor Chira Pitha | Bihu |
| BIHU_021 | Petore Juruni Bhat | Bihu |
| BIHU_026 | Bohag Ahil Buli | Bihu |
| BIHU_029 | Morone Phulonit | Bihu |
| BIHU_030 | Nila Nila Akhore | Bihu |
| BIHU_035 | Bohagi | Bihu |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Dhamail |
| DHAMAIL_027 | অচেনা এক রূপবতী | Dhamail |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Dhamail |
| DHAMAIL_030 | জানো নি গো সই | Dhamail |
| GARBA_005 | Amba Aavo To Ramiye | Garba |
| GARBA_007 | Maa No Garbo Re | Garba |
| GARBA_009 | Aavya Maa Na Norta | Garba |
| GARBA_014 | Garba No Rang Lagiyo | Garba |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba |
| GARBA_017 | Halaji Tara Hath Vakhanu | Garba |
| GARBA_019 | Pankhida O Pankhida | Garba |
| GARBA_026 | Kaho Poonam Na Chand | Garba |
| GHOOMAR_002 | Gangore/Gangaur | Ghoomar |
| GHOOMAR_004 | More Bole Re | Ghoomar |
| GHOOMAR_006 | Talariya Magriya | Ghoomar |
| GHOOMAR_007 | Panihari Ji | Ghoomar |
| GHOOMAR_012 | Chirmi | Ghoomar |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar |
| GHOOMAR_014 | Dal Badal | Ghoomar |
| GHOOMAR_016 | Jeero | Ghoomar |
| GHOOMAR_018 | Taran Ri Chunadi | Ghoomar |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar |
| GHOOMAR_030 | Dar Lanjo | Ghoomar |
| JHUMUR_0015 | Jhatka | Jhumur |
| JHUMUR_0018 | Koromoti | Jhumur |
| JHUMUR_0019 | Sonamoni | Jhumur |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur |
| JHUMUR_0022 | Deewana | Jhumur |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Jhumur |
| JHUMUR_0024 | Laal Pari | Jhumur |
| JHUMUR_0025 | Tore Bina | Jhumur |
| JHUMUR_0027 | The Jhumur Song | Jhumur |
| JHUMUR_0028 | Genda Phool | Jhumur |
| JHUMUR_0030 | Goya | Jhumur |
| JHUMUR_004 | Sajanigo | Jhumur |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Lavani |
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand |
| MAAND_011 | Taniya Resham Ki | Maand |
| MAAND_017 | Apne Desh Ri Aajadi | Maand |
| MAAND_019 | Raag Maand Bandish | Maand |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand |
| MAAND_037 | Padharo Mhare Mehal | Maand |
| MAAND_039 | Mahara Sajan Base | Maand |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri |
| SAMBALPURI_010 | Rani Guri | Sambalpuri |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri |
| SAMBALPURI_018 | Prem Paga | Sambalpuri |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri |
| SAMBALPURI_031 | Rangabati | Sambalpuri |

### Ahir Bhairav
| Song ID | Song | Genre |
|---|---|---|
| BAUL_019 | Aami Ek Pagal Baul | Baul |
| BHANGRA_004 | Wang Wargi Kudi | Bhangra |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra |
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra |
| BIHU_005 | Kinu Sawonire Sala | Bihu |
| BIHU_009 | O Nahor | Bihu |
| BIHU_013 | Ki Sai Aso | Bihu |
| BIHU_014 | Bihureno Birina | Bihu |
| BIHU_020 | Dhansiri Pare Hoi | Bihu |
| BIHU_025 | Senai o Tumake Bisari | Bihu |
| BIHU_026 | Bohag Ahil Buli | Bihu |
| BIHU_027 | Tumi Nasoni | Bihu |
| BIHU_028 | Xunjone Nuxudhiba | Bihu |
| BIHU_029 | Morone Phulonit | Bihu |
| BIHU_031 | Kahili puwate | Bihu |
| BIHU_034 | Gabhoru Bihu | Bihu |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail |
| DHAMAIL_027 | অচেনা এক রূপবতী | Dhamail |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba |
| GHOOMAR_009 | Gorband | Ghoomar |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar |
| GHOOMAR_020 | Kangasiyo | Ghoomar |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar |
| JHUMUR_0011 | Barire Simare | Jhumur |
| JHUMUR_0013 | Jhuli Jhule | Jhumur |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur |
| LAVANI_017 | Kashyala Lavato | Lavani |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand |
| MAAND_018 | Kesariya Balam Aao Ni | Maand |
| MAAND_021 | Chaudhary | Maand |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand |
| MAAND_026 | Darude Ro Rog Lagiyo | Maand |
| MAAND_029 | Kalali | Maand |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri |
| SAMBALPURI_008 | Hae Go | Sambalpuri |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri |
| SAMBALPURI_018 | Prem Paga | Sambalpuri |
| SAMBALPURI_021 | Badnam | Sambalpuri |
| SAMBALPURI_022 | Dil Bhitre Lekhideli | Sambalpuri |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri |

### Alhaiya Bilaval
| Song ID | Song | Genre |
|---|---|---|
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul |
| BAUL_005 | Amar Aei Rudir Dhara | Baul |
| BAUL_011 | Chor Poreche Babur Bagane | Baul |
| BHANGRA_001 | Tera Yaar Bolda | Bhangra |
| BHANGRA_003 | Dupatta Tera Satrang Da | Bhangra |
| BHANGRA_006 | Munda Ki Mangda | Bhangra |
| BHANGRA_010 | Dardi Rab Rab Kardi | Bhangra |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra |
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra |
| BHANGRA_016 | Mitran Ne Mundri Te | Bhangra |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra |
| BHANGRA_021 | Kinni Sohni | Bhangra |
| BHANGRA_023 | Bhaina Nachiyaan | Bhangra |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Bhangra |
| BIHU_029 | Morone Phulonit | Bihu |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail |
| DHAMAIL_030 | জানো নি গো সই | Dhamail |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Garba |
| GARBA_007 | Maa No Garbo Re | Garba |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba |
| GARBA_021 | He Jagjanani | Garba |
| GARBA_026 | Kaho Poonam Na Chand | Garba |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar |
| GHOOMAR_026 | Ital Peetal | Ghoomar |
| JHUMUR_0025 | Tore Bina | Jhumur |
| LAVANI_017 | Kashyala Lavato | Lavani |
| LAVANI_035 | Chala Jejurila Jau | Lavani |
| MAAND_011 | Taniya Resham Ki | Maand |
| MAAND_012 | Kad Janulee Sasre | Maand |
| MAAND_035 | Rajwadi Mehfil | Maand |
| MAAND_038 | Daru Dakha Ro | Maand |
| SAMBALPURI_010 | Rani Guri | Sambalpuri |
| SAMBALPURI_013 | Ramagaru | Sambalpuri |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri |
| SAMBALPURI_028 | Kulfirani Chocobar | Sambalpuri |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri |

### Asavari
| Song ID | Song | Genre |
|---|---|---|
| BAUL_010 | Sonogo Aian Dada | Baul |
| BAUL_015 | Aami Kothay Pabo Tare | Baul |
| BAUL_022 | Emon Mon Byabsaa | Baul |
| BHANGRA_002 | Jatt Di Pasand | Bhangra |
| BHANGRA_008 | Bolo Ta Ra Ra | Bhangra |
| BHANGRA_009 | Ho Jayegi Balle Balle | Bhangra |
| BHANGRA_010 | Dardi Rab Rab Kardi | Bhangra |
| BHANGRA_013 | Pao Bhangra | Bhangra |
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra |
| BHANGRA_016 | Mitran Ne Mundri Te | Bhangra |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra |
| BHANGRA_023 | Bhaina Nachiyaan | Bhangra |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Bhangra |
| BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Bhangra |
| BIHU_002 | Toi Nagini Ne | Bihu |
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu |
| BIHU_012 | Laje Moru Moru | Bihu |
| BIHU_013 | Ki Sai Aso | Bihu |
| BIHU_015 | Duetar Boraghor | Bihu |
| BIHU_016 | Sangor Chira Pitha | Bihu |
| BIHU_017 | Kopou Phul Phulile | Bihu |
| BIHU_018 | Amona Dhanoke | Bihu |
| BIHU_033 | Dhengkoli | Bihu |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Dhamail |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Dhamail |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail |
| DHAMAIL_030 | জানো নি গো সই | Dhamail |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba |
| GARBA_007 | Maa No Garbo Re | Garba |
| GARBA_008 | Unchi Talavdi Ni Kor | Garba |
| GARBA_017 | Halaji Tara Hath Vakhanu | Garba |
| GARBA_018 | Dholida Dhol Re Vagad | Garba |
| GARBA_026 | Kaho Poonam Na Chand | Garba |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba |
| GHOOMAR_005 | Podina | Ghoomar |
| GHOOMAR_009 | Gorband | Ghoomar |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar |
| GHOOMAR_016 | Jeero | Ghoomar |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar |
| GHOOMAR_030 | Dar Lanjo | Ghoomar |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur |
| JHUMUR_0014 | Kansiya | Jhumur |
| JHUMUR_0016 | Gohona | Jhumur |
| JHUMUR_0017 | Hariya | Jhumur |
| JHUMUR_0025 | Tore Bina | Jhumur |
| JHUMUR_0027 | The Jhumur Song | Jhumur |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur |
| JHUMUR_0030 | Goya | Jhumur |
| LAVANI_002 | Padarawarti Jartaricha | Lavani |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani |
| LAVANI_008 | Disla Ga Bai Disla | Lavani |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani |
| MAAND_006 | Jawai Balha Lagoji | Maand |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand |
| MAAND_023 | Balam | Maand |
| MAAND_026 | Darude Ro Rog Lagiyo | Maand |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri |
| SAMBALPURI_006 | Rasia | Sambalpuri |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri |
| SAMBALPURI_009 | Aukaat | Sambalpuri |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri |
| SAMBALPURI_018 | Prem Paga | Sambalpuri |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri |
| SAMBALPURI_028 | Kulfirani Chocobar | Sambalpuri |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri |

### Bageshree
| Song ID | Song | Genre |
|---|---|---|
| BAUL_021 | Prem Koraa Ki Jwalaa | Baul |
| BAUL_024 | Dil Dariyar Majhete | Baul |
| BAUL_028 | Somoy Gele Sadhon Hobena | Baul |
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra |
| BIHU_019 | O Dhei Oi | Bihu |
| BIHU_036 | Podulit Bihu | Bihu |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Dhamail |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail |
| GHOOMAR_020 | Kangasiyo | Ghoomar |
| JHUMUR_0014 | Kansiya | Jhumur |
| JHUMUR_0016 | Gohona | Jhumur |
| JHUMUR_0018 | Koromoti | Jhumur |
| JHUMUR_002 | Nistei | Jhumur |
| JHUMUR_0028 | Genda Phool | Jhumur |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani |
| LAVANI_012 | Naki Doli Chhaan | Lavani |
| LAVANI_021 | Sola Hazarat | Lavani |
| LAVANI_022 | Hi Natrangi Naar | Lavani |
| LAVANI_023 | Saaman Avjad Majha | Lavani |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani |
| MAAND_006 | Jawai Balha Lagoji | Maand |
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri |
| SAMBALPURI_006 | Rasia | Sambalpuri |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri |
| SAMBALPURI_010 | Rani Guri | Sambalpuri |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri |

### Bahar
| Song ID | Song | Genre |
|---|---|---|
| BAUL_001 | Agun Pani | Baul |
| BAUL_010 | Sonogo Aian Dada | Baul |
| BAUL_029 | Aar Amare Marishne Maa | Baul |
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul |
| BHANGRA_001 | Tera Yaar Bolda | Bhangra |
| BHANGRA_002 | Jatt Di Pasand | Bhangra |
| BHANGRA_003 | Dupatta Tera Satrang Da | Bhangra |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra |
| BHANGRA_013 | Pao Bhangra | Bhangra |
| BIHU_001 | Morom Logai Logai | Bihu |
| BIHU_002 | Toi Nagini Ne | Bihu |
| BIHU_009 | O Nahor | Bihu |
| BIHU_014 | Bihureno Birina | Bihu |
| BIHU_019 | O Dhei Oi | Bihu |
| BIHU_022 | Aag Fale Dekhilo | Bihu |
| BIHU_025 | Senai o Tumake Bisari | Bihu |
| BIHU_028 | Xunjone Nuxudhiba | Bihu |
| BIHU_030 | Nila Nila Akhore | Bihu |
| BIHU_035 | Bohagi | Bihu |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail |
| GARBA_008 | Unchi Talavdi Ni Kor | Garba |
| GARBA_009 | Aavya Maa Na Norta | Garba |
| GARBA_024 | Dam Dam Dakla Vagya | Garba |
| GHOOMAR_003 | Choumaso | Ghoomar |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar |
| GHOOMAR_015 | Samdariyo | Ghoomar |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar |
| JHUMUR_0019 | Sonamoni | Jhumur |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani |
| LAVANI_014 | Wajle Ki Bara | Lavani |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani |
| LAVANI_034 | Reshmachya Reghani | Lavani |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand |
| MAAND_005 | Bai Sa Ra Beera | Maand |
| MAAND_006 | Jawai Balha Lagoji | Maand |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand |
| MAAND_012 | Kad Janulee Sasre | Maand |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand |
| SAMBALPURI_031 | Rangabati | Sambalpuri |

### Barwa
| Song ID | Song | Genre |
|---|---|---|
| BAUL_009 | Kon Ba Zharir | Baul |
| BAUL_017 | Milon Hobe Koto Dine | Baul |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra |
| BHANGRA_023 | Bhaina Nachiyaan | Bhangra |
| BIHU_008 | Porbotor Dhekiya | Bihu |
| BIHU_017 | Kopou Phul Phulile | Bihu |
| DHAMAIL_018 | কত আদরে আদরে | Dhamail |
| GARBA_011 | Sathiya Puravo | Garba |
| GARBA_015 | Hindole Zule Amba | Garba |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur |
| JHUMUR_0015 | Jhatka | Jhumur |
| JHUMUR_0019 | Sonamoni | Jhumur |
| LAVANI_017 | Kashyala Lavato | Lavani |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand |

### Basant
| Song ID | Song | Genre |
|---|---|---|
| BAUL_008 | Tripti | Baul |
| BHANGRA_009 | Ho Jayegi Balle Balle | Bhangra |
| BHANGRA_032 | Jeona Morh | Bhangra |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar |
| JHUMUR_003 | Kuilu Kuilu | Jhumur |
| JHUMUR_004 | Sajanigo | Jhumur |
| JHUMUR_007 | Kale Gori Muke | Jhumur |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani |
| LAVANI_015 | Apsara Aali | Lavani |
| LAVANI_034 | Reshmachya Reghani | Lavani |
| MAAND_017 | Apne Desh Ri Aajadi | Maand |
| MAAND_018 | Kesariya Balam Aao Ni | Maand |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand |
| MAAND_035 | Rajwadi Mehfil | Maand |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri |

### Basant Mukhari
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_032 | Jeona Morh | Bhangra |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba |
| GARBA_022 | Chotile Dakla Vagya | Garba |
| GARBA_024 | Dam Dam Dakla Vagya | Garba |
| GHOOMAR_001 | Ghoomar | Ghoomar |
| GHOOMAR_004 | More Bole Re | Ghoomar |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar |
| JHUMUR_003 | Kuilu Kuilu | Jhumur |
| JHUMUR_007 | Kale Gori Muke | Jhumur |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani |

### Bhairav
| Song ID | Song | Genre |
|---|---|---|
| BAUL_020 | Jemon Beni Temni Rabey | Baul |
| BAUL_027 | Pre Loye Jao Amay | Baul |
| BHANGRA_032 | Jeona Morh | Bhangra |
| BIHU_034 | Gabhoru Bihu | Bihu |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail |
| GARBA_025 | Sonala Vatakadi | Garba |
| GHOOMAR_002 | Gangore/Gangaur | Ghoomar |
| GHOOMAR_004 | More Bole Re | Ghoomar |
| GHOOMAR_005 | Podina | Ghoomar |
| GHOOMAR_006 | Talariya Magriya | Ghoomar |
| GHOOMAR_009 | Gorband | Ghoomar |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar |
| GHOOMAR_020 | Kangasiyo | Ghoomar |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar |
| GHOOMAR_025 | Lehariyo | Ghoomar |
| GHOOMAR_030 | Dar Lanjo | Ghoomar |
| JHUMUR_0027 | The Jhumur Song | Jhumur |
| JHUMUR_003 | Kuilu Kuilu | Jhumur |
| JHUMUR_005 | Sol Sokhi | Jhumur |
| LAVANI_022 | Hi Natrangi Naar | Lavani |
| MAAND_012 | Kad Janulee Sasre | Maand |
| SAMBALPURI_031 | Rangabati | Sambalpuri |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri |

### Bhairav Bahar
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_006 | Munda Ki Mangda | Bhangra |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba |
| MAAND_005 | Bai Sa Ra Beera | Maand |

### Bhairavi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_001 | Agun Pani | Baul |
| BAUL_004 | Ananda Lahari | Baul |
| BAUL_005 | Amar Aei Rudir Dhara | Baul |
| BAUL_008 | Tripti | Baul |
| BAUL_011 | Chor Poreche Babur Bagane | Baul |
| BAUL_014 | Dekhechhi Rup-Sagare | Baul |
| BAUL_015 | Aami Kothay Pabo Tare | Baul |
| BAUL_019 | Aami Ek Pagal Baul | Baul |
| BHANGRA_013 | Pao Bhangra | Bhangra |
| BHANGRA_029 | Putt Jattan De | Bhangra |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra |
| BIHU_019 | O Dhei Oi | Bihu |
| BIHU_020 | Dhansiri Pare Hoi | Bihu |
| BIHU_021 | Petore Juruni Bhat | Bihu |
| BIHU_022 | Aag Fale Dekhilo | Bihu |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani |
| LAVANI_014 | Wajle Ki Bara | Lavani |
| LAVANI_022 | Hi Natrangi Naar | Lavani |
| LAVANI_035 | Chala Jejurila Jau | Lavani |
| MAAND_012 | Kad Janulee Sasre | Maand |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand |
| MAAND_021 | Chaudhary | Maand |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand |
| MAAND_029 | Kalali | Maand |
| MAAND_032 | Alija Dhola Ekladi | Maand |
| MAAND_034 | Mhari Sajni Ae | Maand |
| MAAND_035 | Rajwadi Mehfil | Maand |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri |

### Bhatiyar
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_007 | Tunak Tunak Tun | Bhangra |
| BIHU_020 | Dhansiri Pare Hoi | Bihu |
| BIHU_029 | Morone Phulonit | Bihu |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail |
| GHOOMAR_027 | Moriya | Ghoomar |
| JHUMUR_001 | Kati Khane Bole | Jhumur |
| JHUMUR_005 | Sol Sokhi | Jhumur |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand |
| MAAND_033 | Mumal | Maand |
| MAAND_039 | Mahara Sajan Base | Maand |
| SAMBALPURI_009 | Aukaat | Sambalpuri |

### Bhimpalasi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_010 | Sonogo Aian Dada | Baul |
| BAUL_017 | Milon Hobe Koto Dine | Baul |
| BHANGRA_020 | Chal Hun | Bhangra |
| BHANGRA_028 | Jugni | Bhangra |
| BIHU_001 | Morom Logai Logai | Bihu |
| BIHU_009 | O Nahor | Bihu |
| BIHU_011 | Eijon Korenu Nasoni | Bihu |
| BIHU_026 | Bohag Ahil Buli | Bihu |
| BIHU_033 | Dhengkoli | Bihu |
| DHAMAIL_011 | বল গো বল সখি | Dhamail |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba |
| GARBA_018 | Dholida Dhol Re Vagad | Garba |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar |
| JHUMUR_0022 | Deewana | Jhumur |
| JHUMUR_0026 | Koromake Akhora | Jhumur |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani |
| LAVANI_015 | Apsara Aali | Lavani |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani |
| MAAND_027 | Kesariya Balam Padharo | Maand |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand |
| MAAND_032 | Alija Dhola Ekladi | Maand |
| MAAND_033 | Mumal | Maand |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri |
| SAMBALPURI_006 | Rasia | Sambalpuri |
| SAMBALPURI_009 | Aukaat | Sambalpuri |
| SAMBALPURI_027 | Feel My Love | Sambalpuri |

### Bhinna Shadja
| Song ID | Song | Genre |
|---|---|---|
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba |
| JHUMUR_001 | Kati Khane Bole | Jhumur |
| JHUMUR_003 | Kuilu Kuilu | Jhumur |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur |
| LAVANI_015 | Apsara Aali | Lavani |

### Bhoopali
| Song ID | Song | Genre |
|---|---|---|
| BAUL_017 | Milon Hobe Koto Dine | Baul |
| BAUL_027 | Pre Loye Jao Amay | Baul |
| BHANGRA_001 | Tera Yaar Bolda | Bhangra |
| BHANGRA_002 | Jatt Di Pasand | Bhangra |
| BIHU_012 | Laje Moru Moru | Bihu |
| BIHU_013 | Ki Sai Aso | Bihu |
| BIHU_015 | Duetar Boraghor | Bihu |
| BIHU_016 | Sangor Chira Pitha | Bihu |
| BIHU_017 | Kopou Phul Phulile | Bihu |
| BIHU_023 | Aghunor Potharot | Bihu |
| BIHU_024 | Ebela Hale Bau | Bihu |
| BIHU_026 | Bohag Ahil Buli | Bihu |
| BIHU_028 | Xunjone Nuxudhiba | Bihu |
| BIHU_030 | Nila Nila Akhore | Bihu |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail |
| GARBA_015 | Hindole Zule Amba | Garba |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba |
| GARBA_018 | Dholida Dhol Re Vagad | Garba |
| GHOOMAR_003 | Choumaso | Ghoomar |
| GHOOMAR_006 | Talariya Magriya | Ghoomar |
| GHOOMAR_007 | Panihari Ji | Ghoomar |
| GHOOMAR_012 | Chirmi | Ghoomar |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar |
| GHOOMAR_030 | Dar Lanjo | Ghoomar |
| JHUMUR_0015 | Jhatka | Jhumur |
| JHUMUR_0017 | Hariya | Jhumur |
| JHUMUR_0020 | Chaliya Chhodi | Jhumur |
| JHUMUR_0025 | Tore Bina | Jhumur |
| LAVANI_013 | Aai Majhya Lagnachi | Lavani |
| LAVANI_014 | Wajle Ki Bara | Lavani |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Lavani |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand |
| MAAND_027 | Kesariya Balam Padharo | Maand |
| SAMBALPURI_010 | Rani Guri | Sambalpuri |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri |

### Bhupal Todi
| Song ID | Song | Genre |
|---|---|---|
| BIHU_018 | Amona Dhanoke | Bihu |
| DHAMAIL_018 | কত আদরে আদরে | Dhamail |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba |
| MAAND_037 | Padharo Mhare Mehal | Maand |

### Bibhas
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_001 | Tera Yaar Bolda | Bhangra |
| BHANGRA_003 | Dupatta Tera Satrang Da | Bhangra |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra |
| GHOOMAR_001 | Ghoomar | Ghoomar |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar |
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand |
| MAAND_037 | Padharo Mhare Mehal | Maand |

### Bihag
| Song ID | Song | Genre |
|---|---|---|
| BAUL_022 | Emon Mon Byabsaa | Baul |
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul |
| BIHU_034 | Gabhoru Bihu | Bihu |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail |
| DHAMAIL_021 | জলের ঘাটে | Dhamail |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba |
| GARBA_029 | Pavana Dungre | Garba |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani |
| LAVANI_012 | Naki Doli Chhaan | Lavani |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani |
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri |

### Bilaskhani Todi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_029 | Aar Amare Marishne Maa | Baul |
| BHANGRA_007 | Tunak Tunak Tun | Bhangra |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra |
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Bhangra |
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu |
| BIHU_030 | Nila Nila Akhore | Bihu |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail |
| GARBA_002 | Madi Tara Mandiriye Ghantarav Thay | Garba |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani |
| LAVANI_021 | Sola Hazarat | Lavani |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani |
| MAAND_004 | Ghoomar | Maand |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand |
| MAAND_014 | The To Piyo Sadarudi | Maand |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand |
| MAAND_033 | Mumal | Maand |

### Bilaval
| Song ID | Song | Genre |
|---|---|---|
| BAUL_012 | Golemale Gole Male | Baul |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra |
| BHANGRA_025 | Soho Road Utey | Bhangra |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra |
| BIHU_004 | Tumake Logot | Bihu |
| BIHU_021 | Petore Juruni Bhat | Bihu |
| BIHU_022 | Aag Fale Dekhilo | Bihu |
| BIHU_025 | Senai o Tumake Bisari | Bihu |
| BIHU_032 | Rohare O Rohedoi | Bihu |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail |
| GARBA_015 | Hindole Zule Amba | Garba |
| GARBA_025 | Sonala Vatakadi | Garba |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur |
| JHUMUR_0018 | Koromoti | Jhumur |
| JHUMUR_0026 | Koromake Akhora | Jhumur |
| JHUMUR_0028 | Genda Phool | Jhumur |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani |
| MAAND_019 | Raag Maand Bandish | Maand |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri |
| SAMBALPURI_006 | Rasia | Sambalpuri |
| SAMBALPURI_008 | Hae Go | Sambalpuri |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri |

### Darbari
| Song ID | Song | Genre |
|---|---|---|
| BAUL_001 | Agun Pani | Baul |
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul |
| BAUL_003 | Kalkatou Mon Heshekeley | Baul |
| BAUL_004 | Ananda Lahari | Baul |
| BAUL_006 | Hari Balo Re | Baul |
| BAUL_007 | Gour Lila Bazare | Baul |
| BAUL_016 | Shesher Deene Sejon Bine | Baul |
| BAUL_019 | Aami Ek Pagal Baul | Baul |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Baul |
| BHANGRA_001 | Tera Yaar Bolda | Bhangra |
| BHANGRA_002 | Jatt Di Pasand | Bhangra |
| BHANGRA_003 | Dupatta Tera Satrang Da | Bhangra |
| BHANGRA_004 | Wang Wargi Kudi | Bhangra |
| BHANGRA_009 | Ho Jayegi Balle Balle | Bhangra |
| BHANGRA_010 | Dardi Rab Rab Kardi | Bhangra |
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra |
| BHANGRA_016 | Mitran Ne Mundri Te | Bhangra |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra |
| BHANGRA_020 | Chal Hun | Bhangra |
| BHANGRA_021 | Kinni Sohni | Bhangra |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra |
| BHANGRA_025 | Soho Road Utey | Bhangra |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Bhangra |
| BHANGRA_027 | Jodi (Big Day Party) ft. Panjabi MC | Bhangra |
| BHANGRA_029 | Putt Jattan De | Bhangra |
| BIHU_004 | Tumake Logot | Bihu |
| BIHU_005 | Kinu Sawonire Sala | Bihu |
| BIHU_007 | Phool Phulise | Bihu |
| BIHU_012 | Laje Moru Moru | Bihu |
| BIHU_015 | Duetar Boraghor | Bihu |
| BIHU_023 | Aghunor Potharot | Bihu |
| BIHU_024 | Ebela Hale Bau | Bihu |
| BIHU_027 | Tumi Nasoni | Bihu |
| BIHU_029 | Morone Phulonit | Bihu |
| BIHU_031 | Kahili puwate | Bihu |
| BIHU_032 | Rohare O Rohedoi | Bihu |
| BIHU_034 | Gabhoru Bihu | Bihu |
| BIHU_036 | Podulit Bihu | Bihu |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dhamail |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail |
| DHAMAIL_011 | বল গো বল সখি | Dhamail |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Dhamail |
| DHAMAIL_019 | জলে গিয়েছিলাম সই | Dhamail |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail |
| DHAMAIL_021 | জলের ঘাটে | Dhamail |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Garba |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba |
| GARBA_007 | Maa No Garbo Re | Garba |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba |
| GARBA_015 | Hindole Zule Amba | Garba |
| GARBA_017 | Halaji Tara Hath Vakhanu | Garba |
| GARBA_029 | Pavana Dungre | Garba |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar |
| GHOOMAR_013 | Hichki (Aave Hichki) | Ghoomar |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar |
| GHOOMAR_025 | Lehariyo | Ghoomar |
| GHOOMAR_026 | Ital Peetal | Ghoomar |
| JHUMUR_0011 | Barire Simare | Jhumur |
| JHUMUR_0014 | Kansiya | Jhumur |
| JHUMUR_0016 | Gohona | Jhumur |
| JHUMUR_0017 | Hariya | Jhumur |
| JHUMUR_0026 | Koromake Akhora | Jhumur |
| JHUMUR_0028 | Genda Phool | Jhumur |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur |
| JHUMUR_003 | Kuilu Kuilu | Jhumur |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur |
| LAVANI_017 | Kashyala Lavato | Lavani |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani |
| LAVANI_035 | Chala Jejurila Jau | Lavani |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand |
| MAAND_012 | Kad Janulee Sasre | Maand |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand |
| MAAND_023 | Balam | Maand |
| MAAND_026 | Darude Ro Rog Lagiyo | Maand |
| MAAND_034 | Mhari Sajni Ae | Maand |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri |
| SAMBALPURI_008 | Hae Go | Sambalpuri |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri |
| SAMBALPURI_018 | Prem Paga | Sambalpuri |
| SAMBALPURI_027 | Feel My Love | Sambalpuri |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri |

### Deepak
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_007 | Tunak Tunak Tun | Bhangra |
| BHANGRA_011 | Saade Dil Te Chhuriyan Chaliyan | Bhangra |
| BIHU_022 | Aag Fale Dekhilo | Bihu |
| JHUMUR_005 | Sol Sokhi | Jhumur |

### Desh
| Song ID | Song | Genre |
|---|---|---|
| BAUL_009 | Kon Ba Zharir | Baul |
| BAUL_017 | Milon Hobe Koto Dine | Baul |
| BAUL_020 | Jemon Beni Temni Rabey | Baul |
| BAUL_024 | Dil Dariyar Majhete | Baul |
| BAUL_026 | Sokhi Jamuna Jol | Baul |
| BAUL_029 | Aar Amare Marishne Maa | Baul |
| BHANGRA_008 | Bolo Ta Ra Ra | Bhangra |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra |
| BHANGRA_024 | Apna Sangeet Vajeh Apna Sangeet | Bhangra |
| BHANGRA_025 | Soho Road Utey | Bhangra |
| BHANGRA_028 | Jugni | Bhangra |
| BIHU_005 | Kinu Sawonire Sala | Bihu |
| BIHU_006 | Dikhow Noi Eribo | Bihu |
| BIHU_007 | Phool Phulise | Bihu |
| BIHU_008 | Porbotor Dhekiya | Bihu |
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu |
| BIHU_018 | Amona Dhanoke | Bihu |
| BIHU_019 | O Dhei Oi | Bihu |
| BIHU_023 | Aghunor Potharot | Bihu |
| BIHU_036 | Podulit Bihu | Bihu |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail |
| DHAMAIL_011 | বল গো বল সখি | Dhamail |
| DHAMAIL_018 | কত আদরে আদরে | Dhamail |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba |
| GARBA_009 | Aavya Maa Na Norta | Garba |
| GARBA_019 | Pankhida O Pankhida | Garba |
| GARBA_021 | He Jagjanani | Garba |
| GARBA_022 | Chotile Dakla Vagya | Garba |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba |
| GHOOMAR_020 | Kangasiyo | Ghoomar |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar |
| JHUMUR_0019 | Sonamoni | Jhumur |
| JHUMUR_0022 | Deewana | Jhumur |
| JHUMUR_0024 | Laal Pari | Jhumur |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani |
| LAVANI_009 | Chabidar Chabi | Lavani |
| LAVANI_020 | Themba Thembana Kelai | Lavani |
| LAVANI_021 | Sola Hazarat | Lavani |
| LAVANI_023 | Saaman Avjad Majha | Lavani |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani |
| LAVANI_028 | Navi Navi Navari Mee | Lavani |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani |
| LAVANI_032 | Aase Wajwa Ki | Lavani |
| LAVANI_035 | Chala Jejurila Jau | Lavani |
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand |
| MAAND_003 | Moomal | Maand |
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand |
| MAAND_021 | Chaudhary | Maand |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand |
| MAAND_029 | Kalali | Maand |
| MAAND_032 | Alija Dhola Ekladi | Maand |
| MAAND_039 | Mahara Sajan Base | Maand |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri |
| SAMBALPURI_006 | Rasia | Sambalpuri |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri |
| SAMBALPURI_013 | Ramagaru | Sambalpuri |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri |
| SAMBALPURI_017 | Chandir Paenjhal | Sambalpuri |
| SAMBALPURI_027 | Feel My Love | Sambalpuri |

### Desi
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_021 | Kinni Sohni | Bhangra |
| BHANGRA_029 | Putt Jattan De | Bhangra |
| BIHU_005 | Kinu Sawonire Sala | Bihu |
| BIHU_007 | Phool Phulise | Bihu |
| BIHU_008 | Porbotor Dhekiya | Bihu |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail |
| GARBA_029 | Pavana Dungre | Garba |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar |
| JHUMUR_0013 | Jhuli Jhule | Jhumur |
| JHUMUR_004 | Sajanigo | Jhumur |
| LAVANI_019 | Jhampar Majha Dheel | Lavani |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri |

### Durga
| Song ID | Song | Genre |
|---|---|---|
| BAUL_022 | Emon Mon Byabsaa | Baul |
| GARBA_014 | Garba No Rang Lagiyo | Garba |
| GARBA_019 | Pankhida O Pankhida | Garba |
| GARBA_022 | Chotile Dakla Vagya | Garba |
| GHOOMAR_015 | Samdariyo | Ghoomar |
| GHOOMAR_026 | Ital Peetal | Ghoomar |
| LAVANI_013 | Aai Majhya Lagnachi | Lavani |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani |
| MAAND_014 | The To Piyo Sadarudi | Maand |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri |

### Gauri
| Song ID | Song | Genre |
|---|---|---|
| BAUL_011 | Chor Poreche Babur Bagane | Baul |
| BAUL_021 | Prem Koraa Ki Jwalaa | Baul |
| BAUL_022 | Emon Mon Byabsaa | Baul |
| BHANGRA_019 | Kurri Garam Jayee | Bhangra |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail |
| GARBA_024 | Dam Dam Dakla Vagya | Garba |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba |
| LAVANI_008 | Disla Ga Bai Disla | Lavani |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani |
| LAVANI_034 | Reshmachya Reghani | Lavani |
| MAAND_006 | Jawai Balha Lagoji | Maand |
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand |
| MAAND_018 | Kesariya Balam Aao Ni | Maand |
| MAAND_038 | Daru Dakha Ro | Maand |

### Gujari Todi
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra |
| BHANGRA_018 | Tutak Tutak Tutiya | Bhangra |
| BHANGRA_020 | Chal Hun | Bhangra |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail |
| GARBA_006 | Sonal Garbo Shire | Garba |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar |

### Hemant
| Song ID | Song | Genre |
|---|---|---|
| GARBA_008 | Unchi Talavdi Ni Kor | Garba |
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand |

### Hindol
| Song ID | Song | Genre |
|---|---|---|
| BIHU_003 | Kokalote Tongali | Bihu |
| BIHU_005 | Kinu Sawonire Sala | Bihu |
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu |
| BIHU_031 | Kahili puwate | Bihu |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri |

### Jaunpuri
| Song ID | Song | Genre |
|---|---|---|
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Baul |
| BHANGRA_006 | Munda Ki Mangda | Bhangra |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail |
| GARBA_014 | Garba No Rang Lagiyo | Garba |
| GARBA_019 | Pankhida O Pankhida | Garba |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba |
| GHOOMAR_018 | Taran Ri Chunadi | Ghoomar |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand |
| MAAND_003 | Moomal | Maand |

### Jhinjhoti
| Song ID | Song | Genre |
|---|---|---|
| BAUL_012 | Golemale Gole Male | Baul |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul |
| BIHU_015 | Duetar Boraghor | Bihu |

### Kafi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_007 | Gour Lila Bazare | Baul |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul |
| GARBA_008 | Unchi Talavdi Ni Kor | Garba |
| GARBA_021 | He Jagjanani | Garba |
| GARBA_029 | Pavana Dungre | Garba |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani |
| LAVANI_017 | Kashyala Lavato | Lavani |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand |
| MAAND_038 | Daru Dakha Ro | Maand |

### Kedar
| Song ID | Song | Genre |
|---|---|---|
| BAUL_015 | Aami Kothay Pabo Tare | Baul |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Baul |
| BHANGRA_009 | Ho Jayegi Balle Balle | Bhangra |
| BHANGRA_019 | Kurri Garam Jayee | Bhangra |
| BIHU_035 | Bohagi | Bihu |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail |
| DHAMAIL_021 | জলের ঘাটে | Dhamail |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail |
| GARBA_025 | Sonala Vatakadi | Garba |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar |
| GHOOMAR_025 | Lehariyo | Ghoomar |
| JHUMUR_0014 | Kansiya | Jhumur |
| JHUMUR_0017 | Hariya | Jhumur |
| JHUMUR_0018 | Koromoti | Jhumur |
| JHUMUR_0028 | Genda Phool | Jhumur |
| LAVANI_002 | Padarawarti Jartaricha | Lavani |
| LAVANI_008 | Disla Ga Bai Disla | Lavani |
| LAVANI_012 | Naki Doli Chhaan | Lavani |
| LAVANI_028 | Navi Navi Navari Mee | Lavani |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani |
| SAMBALPURI_005 | Lal Sadhi | Sambalpuri |
| SAMBALPURI_021 | Badnam | Sambalpuri |
| SAMBALPURI_030 | Sundri Janha | Sambalpuri |
| SAMBALPURI_031 | Rangabati | Sambalpuri |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri |

### Khamaj
| Song ID | Song | Genre |
|---|---|---|
| BAUL_004 | Ananda Lahari | Baul |
| BAUL_012 | Golemale Gole Male | Baul |
| BAUL_020 | Jemon Beni Temni Rabey | Baul |
| BAUL_028 | Somoy Gele Sadhon Hobena | Baul |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra |
| BHANGRA_025 | Soho Road Utey | Bhangra |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra |
| BIHU_002 | Toi Nagini Ne | Bihu |
| BIHU_003 | Kokalote Tongali | Bihu |
| BIHU_014 | Bihureno Birina | Bihu |
| BIHU_018 | Amona Dhanoke | Bihu |
| BIHU_032 | Rohare O Rohedoi | Bihu |
| BIHU_033 | Dhengkoli | Bihu |
| BIHU_034 | Gabhoru Bihu | Bihu |
| BIHU_035 | Bohagi | Bihu |
| BIHU_036 | Podulit Bihu | Bihu |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail |
| DHAMAIL_013 | শুনোগো শুনোগো সখি | Dhamail |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail |
| DHAMAIL_023 | Tara Lowo Go Jol | Dhamail |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba |
| GARBA_024 | Dam Dam Dakla Vagya | Garba |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba |
| GHOOMAR_001 | Ghoomar | Ghoomar |
| GHOOMAR_004 | More Bole Re | Ghoomar |
| GHOOMAR_006 | Talariya Magriya | Ghoomar |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar |
| GHOOMAR_011 | Hivde Ro Haar | Ghoomar |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar |
| GHOOMAR_025 | Lehariyo | Ghoomar |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar |
| JHUMUR_0013 | Jhuli Jhule | Jhumur |
| JHUMUR_0016 | Gohona | Jhumur |
| JHUMUR_0018 | Koromoti | Jhumur |
| JHUMUR_0020 | Chaliya Chhodi | Jhumur |
| JHUMUR_0026 | Koromake Akhora | Jhumur |
| LAVANI_004 | Kasa Kay Patil Bara Hay Ka | Lavani |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani |
| LAVANI_028 | Navi Navi Navari Mee | Lavani |
| LAVANI_034 | Reshmachya Reghani | Lavani |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand |
| MAAND_010 | Badila Yahi Rahiyo Saa | Maand |
| MAAND_011 | Taniya Resham Ki | Maand |
| MAAND_013 | Le Dekh Nangi Melo | Maand |
| MAAND_018 | Kesariya Balam Aao Ni | Maand |
| MAAND_019 | Raag Maand Bandish | Maand |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand |
| MAAND_027 | Kesariya Balam Padharo | Maand |
| MAAND_038 | Daru Dakha Ro | Maand |
| SAMBALPURI_011 | Gidja Gija Re | Sambalpuri |
| SAMBALPURI_029 | Jhupa Jhupa Baul Jhupa | Sambalpuri |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri |

### Komal Bhairav
| Song ID | Song | Genre |
|---|---|---|
| BIHU_011 | Eijon Korenu Nasoni | Bihu |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail |
| GARBA_006 | Sonal Garbo Shire | Garba |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur |
| LAVANI_003 | Solava Varis Dhokyacha | Lavani |
| MAAND_027 | Kesariya Balam Padharo | Maand |
| MAAND_037 | Padharo Mhare Mehal | Maand |

### Lalit
| Song ID | Song | Genre |
|---|---|---|
| BAUL_009 | Kon Ba Zharir | Baul |
| BIHU_020 | Dhansiri Pare Hoi | Bihu |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail |
| GARBA_003 | Ghor Andhari Re Rataldi Maa | Garba |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba |
| GARBA_019 | Pankhida O Pankhida | Garba |
| GHOOMAR_027 | Moriya | Ghoomar |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Jhumur |
| JHUMUR_0026 | Koromake Akhora | Jhumur |
| JHUMUR_005 | Sol Sokhi | Jhumur |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani |
| MAAND_002 | Moray Aaj Aaye Saiyan | Maand |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand |

### Madhuvanti
| Song ID | Song | Genre |
|---|---|---|
| BAUL_001 | Agun Pani | Baul |
| BAUL_003 | Kalkatou Mon Heshekeley | Baul |
| BAUL_016 | Shesher Deene Sejon Bine | Baul |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra |
| BHANGRA_020 | Chal Hun | Bhangra |
| BHANGRA_026 | Kuriyan Ne Paya Bhangra | Bhangra |
| BIHU_003 | Kokalote Tongali | Bihu |
| BIHU_008 | Porbotor Dhekiya | Bihu |
| BIHU_028 | Xunjone Nuxudhiba | Bihu |
| BIHU_033 | Dhengkoli | Bihu |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dhamail |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail |
| DHAMAIL_003 | বন্ধু অতও যাইবায় গিয়া | Dhamail |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail |
| GARBA_007 | Maa No Garbo Re | Garba |
| GARBA_014 | Garba No Rang Lagiyo | Garba |
| GARBA_027 | Aaj Ni Ghadi Te Raliyamani | Garba |
| GHOOMAR_015 | Samdariyo | Ghoomar |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar |
| GHOOMAR_027 | Moriya | Ghoomar |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar |
| JHUMUR_001 | Kati Khane Bole | Jhumur |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur |
| JHUMUR_0013 | Jhuli Jhule | Jhumur |
| JHUMUR_0024 | Laal Pari | Jhumur |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani |
| MAAND_003 | Moomal | Maand |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand |
| SAMBALPURI_027 | Feel My Love | Sambalpuri |

### Malhar
| Song ID | Song | Genre |
|---|---|---|
| BAUL_004 | Ananda Lahari | Baul |
| BAUL_008 | Tripti | Baul |
| BHANGRA_013 | Pao Bhangra | Bhangra |
| BHANGRA_032 | Jeona Morh | Bhangra |
| BIHU_007 | Phool Phulise | Bihu |
| BIHU_021 | Petore Juruni Bhat | Bihu |
| GARBA_022 | Chotile Dakla Vagya | Garba |
| JHUMUR_007 | Kale Gori Muke | Jhumur |
| LAVANI_005 | Naav Gaav Kashala Pustay | Lavani |
| LAVANI_013 | Aai Majhya Lagnachi | Lavani |
| LAVANI_028 | Navi Navi Navari Mee | Lavani |
| MAAND_013 | Le Dekh Nangi Melo | Maand |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand |
| MAAND_038 | Daru Dakha Ro | Maand |

### Malkauns
| Song ID | Song | Genre |
|---|---|---|
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul |
| BAUL_003 | Kalkatou Mon Heshekeley | Baul |
| BAUL_005 | Amar Aei Rudir Dhara | Baul |
| BAUL_006 | Hari Balo Re | Baul |
| BAUL_008 | Tripti | Baul |
| BAUL_010 | Sonogo Aian Dada | Baul |
| BAUL_015 | Aami Kothay Pabo Tare | Baul |
| BAUL_016 | Shesher Deene Sejon Bine | Baul |
| BAUL_026 | Sokhi Jamuna Jol | Baul |
| BAUL_027 | Pre Loye Jao Amay | Baul |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra |
| BHANGRA_025 | Soho Road Utey | Bhangra |
| BIHU_004 | Tumake Logot | Bihu |
| BIHU_028 | Xunjone Nuxudhiba | Bihu |
| BIHU_031 | Kahili puwate | Bihu |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Dhamail |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba |
| GARBA_006 | Sonal Garbo Shire | Garba |
| GARBA_026 | Kaho Poonam Na Chand | Garba |
| GHOOMAR_001 | Ghoomar | Ghoomar |
| GHOOMAR_004 | More Bole Re | Ghoomar |
| GHOOMAR_005 | Podina | Ghoomar |
| GHOOMAR_009 | Gorband | Ghoomar |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar |
| JHUMUR_001 | Kati Khane Bole | Jhumur |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur |
| JHUMUR_0020 | Chaliya Chhodi | Jhumur |
| JHUMUR_007 | Kale Gori Muke | Jhumur |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani |
| LAVANI_019 | Jhampar Majha Dheel | Lavani |
| LAVANI_023 | Saaman Avjad Majha | Lavani |
| LAVANI_034 | Reshmachya Reghani | Lavani |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand |
| MAAND_029 | Kalali | Maand |
| MAAND_030 | Kesariya Balam | Maand |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri |
| SAMBALPURI_007 | A Mo Dil | Sambalpuri |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri |

### Marva
| Song ID | Song | Genre |
|---|---|---|
| BAUL_004 | Ananda Lahari | Baul |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra |
| BIHU_006 | Dikhow Noi Eribo | Bihu |
| BIHU_009 | O Nahor | Bihu |
| BIHU_014 | Bihureno Birina | Bihu |
| DHAMAIL_011 | বল গো বল সখি | Dhamail |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail |
| GARBA_008 | Unchi Talavdi Ni Kor | Garba |
| GARBA_009 | Aavya Maa Na Norta | Garba |
| GARBA_010 | Khodiyar Maa Nu Holdu | Garba |
| GARBA_025 | Sonala Vatakadi | Garba |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba |
| GHOOMAR_005 | Podina | Ghoomar |
| GHOOMAR_006 | Talariya Magriya | Ghoomar |
| JHUMUR_001 | Kati Khane Bole | Jhumur |
| JHUMUR_0013 | Jhuli Jhule | Jhumur |
| JHUMUR_002 | Nistei | Jhumur |
| JHUMUR_0022 | Deewana | Jhumur |
| JHUMUR_005 | Sol Sokhi | Jhumur |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur |
| LAVANI_002 | Padarawarti Jartaricha | Lavani |
| LAVANI_008 | Disla Ga Bai Disla | Lavani |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani |
| LAVANI_013 | Aai Majhya Lagnachi | Lavani |
| LAVANI_015 | Apsara Aali | Lavani |
| LAVANI_021 | Sola Hazarat | Lavani |
| LAVANI_022 | Hi Natrangi Naar | Lavani |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani |
| LAVANI_028 | Navi Navi Navari Mee | Lavani |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani |
| MAAND_014 | The To Piyo Sadarudi | Maand |
| MAAND_024 | Kuraja Mahra Bhanvr Manae Lav | Maand |
| MAAND_035 | Rajwadi Mehfil | Maand |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri |
| SAMBALPURI_027 | Feel My Love | Sambalpuri |
| SAMBALPURI_028 | Kulfirani Chocobar | Sambalpuri |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri |

### Megh
| Song ID | Song | Genre |
|---|---|---|
| BAUL_007 | Gour Lila Bazare | Baul |
| BAUL_011 | Chor Poreche Babur Bagane | Baul |
| BAUL_026 | Sokhi Jamuna Jol | Baul |
| BAUL_028 | Somoy Gele Sadhon Hobena | Baul |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra |
| BHANGRA_007 | Tunak Tunak Tun | Bhangra |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra |
| BIHU_017 | Kopou Phul Phulile | Bihu |
| BIHU_018 | Amona Dhanoke | Bihu |
| BIHU_019 | O Dhei Oi | Bihu |
| BIHU_024 | Ebela Hale Bau | Bihu |
| BIHU_025 | Senai o Tumake Bisari | Bihu |
| BIHU_035 | Bohagi | Bihu |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail |
| DHAMAIL_027 | অচেনা এক রূপবতী | Dhamail |
| DHAMAIL_028 | কুঞ্জের মাঝে | Dhamail |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba |
| GARBA_005 | Amba Aavo To Ramiye | Garba |
| GARBA_011 | Sathiya Puravo | Garba |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba |
| GHOOMAR_001 | Ghoomar | Ghoomar |
| GHOOMAR_007 | Panihari Ji | Ghoomar |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar |
| GHOOMAR_014 | Dal Badal | Ghoomar |
| GHOOMAR_015 | Samdariyo | Ghoomar |
| GHOOMAR_017 | Loonga Ro Batawo | Ghoomar |
| GHOOMAR_018 | Taran Ri Chunadi | Ghoomar |
| GHOOMAR_027 | Moriya | Ghoomar |
| GHOOMAR_029 | Hathelyan Re Beech | Ghoomar |
| JHUMUR_0010 | Ekadoshi Ratiya | Jhumur |
| JHUMUR_0015 | Jhatka | Jhumur |
| JHUMUR_0023 | Koliye Juge Alai Doyal | Jhumur |
| JHUMUR_0024 | Laal Pari | Jhumur |
| JHUMUR_0027 | The Jhumur Song | Jhumur |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani |
| LAVANI_012 | Naki Doli Chhaan | Lavani |
| LAVANI_032 | Aase Wajwa Ki | Lavani |
| MAAND_005 | Bai Sa Ra Beera | Maand |
| MAAND_029 | Kalali | Maand |
| MAAND_035 | Rajwadi Mehfil | Maand |
| SAMBALPURI_008 | Hae Go | Sambalpuri |
| SAMBALPURI_009 | Aukaat | Sambalpuri |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri |
| SAMBALPURI_028 | Kulfirani Chocobar | Sambalpuri |

### Megh Malhar
| Song ID | Song | Genre |
|---|---|---|
| BAUL_020 | Jemon Beni Temni Rabey | Baul |
| BHANGRA_013 | Pao Bhangra | Bhangra |
| BHANGRA_032 | Jeona Morh | Bhangra |
| BIHU_001 | Morom Logai Logai | Bihu |
| BIHU_021 | Petore Juruni Bhat | Bihu |
| GARBA_006 | Sonal Garbo Shire | Garba |
| GARBA_022 | Chotile Dakla Vagya | Garba |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar |
| GHOOMAR_030 | Dar Lanjo | Ghoomar |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur |
| LAVANI_002 | Padarawarti Jartaricha | Lavani |
| LAVANI_005 | Naav Gaav Kashala Pustay | Lavani |
| LAVANI_008 | Disla Ga Bai Disla | Lavani |
| MAAND_013 | Le Dekh Nangi Melo | Maand |

### Miyan Ki Todi orTodi or Darbari Todi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_029 | Aar Amare Marishne Maa | Baul |
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul |
| BHANGRA_008 | Bolo Ta Ra Ra | Bhangra |
| BHANGRA_010 | Dardi Rab Rab Kardi | Bhangra |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dhamail |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail |
| GARBA_006 | Sonal Garbo Shire | Garba |
| GARBA_014 | Garba No Rang Lagiyo | Garba |
| GARBA_025 | Sonala Vatakadi | Garba |
| GARBA_029 | Pavana Dungre | Garba |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba |
| GHOOMAR_002 | Gangore/Gangaur | Ghoomar |
| GHOOMAR_014 | Dal Badal | Ghoomar |
| GHOOMAR_015 | Samdariyo | Ghoomar |
| GHOOMAR_020 | Kangasiyo | Ghoomar |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar |
| GHOOMAR_024 | Bajudan Ri Loom | Ghoomar |
| GHOOMAR_026 | Ital Peetal | Ghoomar |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur |
| LAVANI_010 | Raya Chala Ghodyavarti Basu | Lavani |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani |
| LAVANI_019 | Jhampar Majha Dheel | Lavani |
| LAVANI_023 | Saaman Avjad Majha | Lavani |
| LAVANI_032 | Aase Wajwa Ki | Lavani |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani |
| MAAND_005 | Bai Sa Ra Beera | Maand |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri |

### Multani
| Song ID | Song | Genre |
|---|---|---|
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail |
| DHAMAIL_016 | যুগল মিলন হইলো গো | Dhamail |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar |
| GHOOMAR_028 | Cham Cham Chake Chundari | Ghoomar |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand |
| MAAND_003 | Moomal | Maand |

### Nat Bhairav
| Song ID | Song | Genre |
|---|---|---|
| BAUL_019 | Aami Ek Pagal Baul | Baul |
| BAUL_028 | Somoy Gele Sadhon Hobena | Baul |
| BIHU_008 | Porbotor Dhekiya | Bihu |
| DHAMAIL_004 | ধামাইল দেও গো চড়চড়ির মা | Dhamail |
| DHAMAIL_005 | বাঁশি বাজগো রইয়া রইয়া | Dhamail |
| DHAMAIL_015 | চান্নি রাইতে আইও কালা | Dhamail |
| DHAMAIL_021 | জলের ঘাটে | Dhamail |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba |
| GHOOMAR_014 | Dal Badal | Ghoomar |
| JHUMUR_0027 | The Jhumur Song | Jhumur |
| JHUMUR_007 | Kale Gori Muke | Jhumur |
| LAVANI_002 | Padarawarti Jartaricha | Lavani |
| LAVANI_005 | Naav Gaav Kashala Pustay | Lavani |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani |
| LAVANI_032 | Aase Wajwa Ki | Lavani |
| MAAND_021 | Chaudhary | Maand |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand |
| SAMBALPURI_019 | Rasa Mandara Mali Kahanra | Sambalpuri |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri |

### Pahadi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_009 | Kon Ba Zharir | Baul |
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Baul |
| BAUL_017 | Milon Hobe Koto Dine | Baul |
| BAUL_029 | Aar Amare Marishne Maa | Baul |
| BHANGRA_008 | Bolo Ta Ra Ra | Bhangra |
| BHANGRA_029 | Putt Jattan De | Bhangra |
| BIHU_007 | Phool Phulise | Bihu |
| DHAMAIL_024 | Jhiko Jhiko Khodi Re | Dhamail |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Dhamail |
| GARBA_016 | Jay Ambe Jagdambe Maa | Garba |
| GARBA_017 | Halaji Tara Hath Vakhanu | Garba |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba |
| GHOOMAR_012 | Chirmi | Ghoomar |
| GHOOMAR_018 | Taran Ri Chunadi | Ghoomar |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar |
| GHOOMAR_026 | Ital Peetal | Ghoomar |
| GHOOMAR_027 | Moriya | Ghoomar |
| JHUMUR_0011 | Barire Simare | Jhumur |
| JHUMUR_002 | Nistei | Jhumur |
| JHUMUR_0029 | Ja Ja Re Mini | Jhumur |
| LAVANI_009 | Chabidar Chabi | Lavani |
| LAVANI_020 | Themba Thembana Kelai | Lavani |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand |
| MAAND_025 | Kurja Mhari Bhanwar Milade | Maand |
| MAAND_028 | Mohe Aai Na Jag Ki Laaj | Maand |
| SAMBALPURI_001 | Nabu Ki Sambalpur | Sambalpuri |
| SAMBALPURI_016 | Anta Halei Halei | Sambalpuri |
| SAMBALPURI_018 | Prem Paga | Sambalpuri |
| SAMBALPURI_020 | A Labanga Lata | Sambalpuri |
| SAMBALPURI_021 | Badnam | Sambalpuri |
| SAMBALPURI_022 | Dil Bhitre Lekhideli | Sambalpuri |
| SAMBALPURI_023 | Bewafa Mita | Sambalpuri |
| SAMBALPURI_024 | Bheti Aibu Sajani Kalekale | Sambalpuri |

### Poorvi
| Song ID | Song | Genre |
|---|---|---|
| BAUL_027 | Pre Loye Jao Amay | Baul |
| DHAMAIL_001 | জাগাইওনা মোরে সই গো | Dhamail |
| DHAMAIL_002 | আর যাবো না জলে | Dhamail |
| DHAMAIL_018 | কত আদরে আদরে | Dhamail |
| DHAMAIL_022 | Bhalo Kare Bajam Re | Dhamail |
| MAAND_008 | Kesariya Balam Awo Ni Padharo | Maand |

### Purba
| Song ID | Song | Genre |
|---|---|---|
| BAUL_001 | Agun Pani | Baul |
| BAUL_005 | Amar Aei Rudir Dhara | Baul |
| BAUL_012 | Golemale Gole Male | Baul |
| BAUL_016 | Shesher Deene Sejon Bine | Baul |
| BAUL_021 | Prem Koraa Ki Jwalaa | Baul |
| BAUL_023 | Manush Bhajo Manusheyi Pujo | Baul |
| BAUL_024 | Dil Dariyar Majhete | Baul |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Baul |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra |
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Bhangra |
| BIHU_010 | Janmoni Xopunote Ahiba | Bihu |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail |
| DHAMAIL_008 | মনের জ্বালায় জ্বলে মরি | Dhamail |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail |
| DHAMAIL_014 | শ্যামরে এখন আর কাঁদিও না | Dhamail |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba |
| GARBA_023 | Jay Aadhyashakti | Garba |
| GARBA_031 | Rumzum Sandhani Aave Re Maa | Garba |
| GHOOMAR_003 | Choumaso | Ghoomar |
| GHOOMAR_005 | Podina | Ghoomar |
| LAVANI_027 | Raya Mala Pavsat Neu Naka | Lavani |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani |
| MAAND_001 | Kesariya Balam Aavo Ni | Maand |
| MAAND_003 | Moomal | Maand |
| MAAND_004 | Ghoomar | Maand |
| MAAND_005 | Bai Sa Ra Beera | Maand |
| MAAND_013 | Le Dekh Nangi Melo | Maand |
| MAAND_016 | Kesariya Balam (DIFF 2015) | Maand |
| MAAND_019 | Raag Maand Bandish | Maand |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand |
| MAAND_027 | Kesariya Balam Padharo | Maand |
| MAAND_030 | Kesariya Balam | Maand |
| MAAND_033 | Mumal | Maand |
| MAAND_039 | Mahara Sajan Base | Maand |

### Puriya
| Song ID | Song | Genre |
|---|---|---|
| BAUL_006 | Hari Balo Re | Baul |
| BAUL_009 | Kon Ba Zharir | Baul |
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Baul |
| BHANGRA_014 | Ambran Tu Ayi Hui Hoor Sohniye | Bhangra |
| BHANGRA_015 | Sariyan Toon Sohni | Bhangra |
| BHANGRA_016 | Mitran Ne Mundri Te | Bhangra |
| BHANGRA_017 | Ambran Tu Ayi Hui Hoor Sohniye (Remix ft. MC Bulla) | Bhangra |
| BHANGRA_019 | Kurri Garam Jayee | Bhangra |
| BHANGRA_023 | Bhaina Nachiyaan | Bhangra |
| BIHU_002 | Toi Nagini Ne | Bihu |
| BIHU_004 | Tumake Logot | Bihu |
| BIHU_009 | O Nahor | Bihu |
| BIHU_012 | Laje Moru Moru | Bihu |
| BIHU_013 | Ki Sai Aso | Bihu |
| BIHU_016 | Sangor Chira Pitha | Bihu |
| BIHU_017 | Kopou Phul Phulile | Bihu |
| BIHU_030 | Nila Nila Akhore | Bihu |
| BIHU_036 | Podulit Bihu | Bihu |
| BIHU_037 | Tok Dekhi Mor Gaa | Bihu |
| DHAMAIL_030 | জানো নি গো সই | Dhamail |
| GARBA_011 | Sathiya Puravo | Garba |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba |
| GARBA_013 | Tara Vina Maa Mane Sunu Lage | Garba |
| GARBA_018 | Dholida Dhol Re Vagad | Garba |
| GARBA_024 | Dam Dam Dakla Vagya | Garba |
| LAVANI_001 | Tujhya Usala Lagal Kolha Ga | Lavani |
| LAVANI_009 | Chabidar Chabi | Lavani |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani |
| LAVANI_012 | Naki Doli Chhaan | Lavani |
| LAVANI_014 | Wajle Ki Bara | Lavani |
| LAVANI_018 | Chhatishi Kavlun Dhara | Lavani |
| LAVANI_019 | Jhampar Majha Dheel | Lavani |
| LAVANI_020 | Themba Thembana Kelai | Lavani |
| LAVANI_021 | Sola Hazarat | Lavani |
| LAVANI_022 | Hi Natrangi Naar | Lavani |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani |
| LAVANI_029 | Ugvali Shukrachi Chandni | Lavani |
| LAVANI_030 | Ghotala Ghotala Zala | Lavani |
| LAVANI_032 | Aase Wajwa Ki | Lavani |
| MAAND_026 | Darude Ro Rog Lagiyo | Maand |
| MAAND_030 | Kesariya Balam | Maand |
| MAAND_034 | Mhari Sajni Ae | Maand |
| MAAND_036 | Thane Samjhaye Sakha Sa | Maand |
| SAMBALPURI_010 | Rani Guri | Sambalpuri |
| SAMBALPURI_025 | Le Le Baula | Sambalpuri |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri |

### Rageshree
| Song ID | Song | Genre |
|---|---|---|
| BAUL_022 | Emon Mon Byabsaa | Baul |
| BAUL_025 | Bahu Punya Phale Krishna Mele | Baul |
| BHANGRA_012 | Na Na Na Na Na Re | Bhangra |
| BHANGRA_019 | Kurri Garam Jayee | Bhangra |
| BHANGRA_021 | Kinni Sohni | Bhangra |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra |
| BIHU_011 | Eijon Korenu Nasoni | Bihu |
| BIHU_014 | Bihureno Birina | Bihu |
| BIHU_022 | Aag Fale Dekhilo | Bihu |
| BIHU_025 | Senai o Tumake Bisari | Bihu |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail |
| GARBA_009 | Aavya Maa Na Norta | Garba |
| GHOOMAR_023 | Naina Ra Lobhi | Ghoomar |
| JHUMUR_004 | Sajanigo | Jhumur |
| LAVANI_005 | Naav Gaav Kashala Pustay | Lavani |
| LAVANI_011 | Pikala Jambhul Todu Naka | Lavani |
| LAVANI_026 | Bai Mazi Karangali Modali | Lavani |
| MAAND_014 | The To Piyo Sadarudi | Maand |
| SAMBALPURI_014 | Deewana Banalu Rani | Sambalpuri |
| SAMBALPURI_021 | Badnam | Sambalpuri |
| SAMBALPURI_031 | Rangabati | Sambalpuri |
| SAMBALPURI_034 | Namo Namo Ago | Sambalpuri |

### Ramkali
| Song ID | Song | Genre |
|---|---|---|
| BAUL_020 | Jemon Beni Temni Rabey | Baul |
| MAAND_037 | Padharo Mhare Mehal | Maand |

### Shree
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_020 | Chal Hun | Bhangra |
| BIHU_020 | Dhansiri Pare Hoi | Bihu |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani |
| SAMBALPURI_033 | Din Gani Gani | Sambalpuri |

### Sohni
| Song ID | Song | Genre |
|---|---|---|
| BAUL_013 | Menokaa Maathaay Dilo Ghomtaa | Baul |
| BIHU_001 | Morom Logai Logai | Bihu |
| BIHU_002 | Toi Nagini Ne | Bihu |
| BIHU_026 | Bohag Ahil Buli | Bihu |
| BIHU_027 | Tumi Nasoni | Bihu |
| BIHU_032 | Rohare O Rohedoi | Bihu |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar |
| GHOOMAR_022 | Kesariya Ro Pag | Ghoomar |
| JHUMUR_0016 | Gohona | Jhumur |
| JHUMUR_002 | Nistei | Jhumur |
| JHUMUR_008 | Tohe Gori Nawa | Jhumur |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur |
| LAVANI_014 | Wajle Ki Bara | Lavani |
| LAVANI_016 | Kashi Mi Jau Mathurechya Bajari | Lavani |
| MAAND_009 | Kesariya Ro Pag (Ghoomar) | Maand |
| MAAND_015 | Kesariya Balam (Desert Slide) | Maand |
| MAAND_018 | Kesariya Balam Aao Ni | Maand |
| SAMBALPURI_009 | Aukaat | Sambalpuri |

### Sorath
| Song ID | Song | Genre |
|---|---|---|
| BAUL_006 | Hari Balo Re | Baul |
| BAUL_008 | Tripti | Baul |
| BAUL_014 | Dekhechhi Rup-Sagare | Baul |
| BAUL_016 | Shesher Deene Sejon Bine | Baul |
| BAUL_019 | Aami Ek Pagal Baul | Baul |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra |
| BHANGRA_028 | Jugni | Bhangra |
| BHANGRA_031 | Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) | Bhangra |
| BIHU_003 | Kokalote Tongali | Bihu |
| BIHU_006 | Dikhow Noi Eribo | Bihu |
| BIHU_023 | Aghunor Potharot | Bihu |
| DHAMAIL_011 | বল গো বল সখি | Dhamail |
| DHAMAIL_012 | ইতা কিতা করে গো মনে | Dhamail |
| DHAMAIL_020 | বন্ধু বিনে প্রাণ বাঁচে না | Dhamail |
| DHAMAIL_025 | Nrittangon Sunamganj | Dhamail |
| DHAMAIL_028 | কুঞ্জের মাঝে | Dhamail |
| DHAMAIL_029 | আজ রাধার কুঞ্জে | Dhamail |
| DHAMAIL_030 | জানো নি গো সই | Dhamail |
| GARBA_001 | Maa Pava Te Gadh Thi Utarya | Garba |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba |
| GARBA_021 | He Jagjanani | Garba |
| GARBA_028 | Lejo Rasiya Re Rumal | Garba |
| GARBA_030 | Momai Na Pare Parevado Dhume Re | Garba |
| JHUMUR_0012 | Mor Dhoni Go | Jhumur |
| JHUMUR_0015 | Jhatka | Jhumur |
| JHUMUR_0019 | Sonamoni | Jhumur |
| JHUMUR_002 | Nistei | Jhumur |
| JHUMUR_0021 | Prem Nadi Tire Tire | Jhumur |
| JHUMUR_0022 | Deewana | Jhumur |
| JHUMUR_0024 | Laal Pari | Jhumur |
| JHUMUR_006 | Aji Ei Suprabhate | Jhumur |
| LAVANI_015 | Apsara Aali | Lavani |
| MAAND_007 | Kesariya Balam Aavo Ni | Maand |
| MAAND_017 | Apne Desh Ri Aajadi | Maand |
| MAAND_019 | Raag Maand Bandish | Maand |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand |
| MAAND_030 | Kesariya Balam | Maand |
| MAAND_034 | Mhari Sajni Ae | Maand |
| SAMBALPURI_002 | Tor Lagi Marijibi | Sambalpuri |
| SAMBALPURI_003 | Sacha Lover | Sambalpuri |
| SAMBALPURI_004 | Pardesia Raja | Sambalpuri |
| SAMBALPURI_008 | Hae Go | Sambalpuri |
| SAMBALPURI_012 | Tui Bewafa | Sambalpuri |
| SAMBALPURI_013 | Ramagaru | Sambalpuri |
| SAMBALPURI_015 | SambalPuri Saiyaan | Sambalpuri |
| SAMBALPURI_032 | Aha Mor Galar Malire | Sambalpuri |

### Yaman Kalyan
| Song ID | Song | Genre |
|---|---|---|
| BAUL_002 | Ja Lolite Anga Kushum Tule | Baul |
| BAUL_003 | Kalkatou Mon Heshekeley | Baul |
| BAUL_006 | Hari Balo Re | Baul |
| BAUL_007 | Gour Lila Bazare | Baul |
| BAUL_010 | Sonogo Aian Dada | Baul |
| BAUL_026 | Sokhi Jamuna Jol | Baul |
| BAUL_030 | Dhonno Dhonno Boli Tare | Baul |
| BHANGRA_022 | Mera Yaar Vajaye Dhol | Bhangra |
| BHANGRA_028 | Jugni | Bhangra |
| BIHU_015 | Duetar Boraghor | Bihu |
| BIHU_027 | Tumi Nasoni | Bihu |
| BIHU_033 | Dhengkoli | Bihu |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail |
| DHAMAIL_017 | বলো গো বলো সই | Dhamail |
| DHAMAIL_018 | কত আদরে আদরে | Dhamail |
| DHAMAIL_021 | জলের ঘাটে | Dhamail |
| GARBA_011 | Sathiya Puravo | Garba |
| GARBA_020 | Tu To Kali Ne Kalyani | Garba |
| GARBA_021 | He Jagjanani | Garba |
| GHOOMAR_008 | Roomal/Rumal | Ghoomar |
| GHOOMAR_012 | Chirmi | Ghoomar |
| GHOOMAR_025 | Lehariyo | Ghoomar |
| JHUMUR_0014 | Kansiya | Jhumur |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani |
| LAVANI_019 | Jhampar Majha Dheel | Lavani |
| LAVANI_023 | Saaman Avjad Majha | Lavani |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani |
| LAVANI_035 | Chala Jejurila Jau | Lavani |
| MAAND_013 | Le Dekh Nangi Melo | Maand |
| MAAND_030 | Kesariya Balam | Maand |
| MAAND_031 | Joori Thaane Na Maare Raad | Maand |
| MAAND_032 | Alija Dhola Ekladi | Maand |
| MAAND_033 | Mumal | Maand |
| MAAND_034 | Mhari Sajni Ae | Maand |
| MAAND_039 | Mahara Sajan Base | Maand |

### Yaman or Kalyan
| Song ID | Song | Genre |
|---|---|---|
| BAUL_007 | Gour Lila Bazare | Baul |
| BAUL_018 | Khanchar Bhitor Achin Pakhi | Baul |
| BAUL_026 | Sokhi Jamuna Jol | Baul |
| BHANGRA_002 | Jatt Di Pasand | Bhangra |
| BHANGRA_005 | Sanu Tedi Tedi Takdi Tu | Bhangra |
| BIHU_016 | Sangor Chira Pitha | Bihu |
| BIHU_024 | Ebela Hale Bau | Bihu |
| DHAMAIL_028 | কুঞ্জের মাঝে | Dhamail |
| GARBA_005 | Amba Aavo To Ramiye | Garba |
| GARBA_012 | Amba Aave Aarasur Vadi | Garba |
| GHOOMAR_003 | Choumaso | Ghoomar |
| GHOOMAR_007 | Panihari Ji | Ghoomar |
| GHOOMAR_010 | Banna Re Bagan Me Jhoola Dalya | Ghoomar |
| GHOOMAR_012 | Chirmi | Ghoomar |
| GHOOMAR_014 | Dal Badal | Ghoomar |
| GHOOMAR_016 | Jeero | Ghoomar |
| JHUMUR_004 | Sajanigo | Jhumur |
| LAVANI_007 | Mala London Foreign Dakhwa | Lavani |
| LAVANI_009 | Chabidar Chabi | Lavani |
| LAVANI_020 | Themba Thembana Kelai | Lavani |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani |
| LAVANI_025 | Bai Mi Patang Udvit Hote | Lavani |
| LAVANI_031 | Raanamandi Ekatach Dultoy Naag | Lavani |
| MAAND_006 | Jawai Balha Lagoji | Maand |
| SAMBALPURI_021 | Badnam | Sambalpuri |

### sarang
| Song ID | Song | Genre |
|---|---|---|
| BAUL_005 | Amar Aei Rudir Dhara | Baul |
| BHANGRA_007 | Tunak Tunak Tun | Bhangra |
| BIHU_023 | Aghunor Potharot | Bihu |
| DHAMAIL_006 | যাইও না রে প্রানো বন্ধু আমারে বুলিয়া | Dhamail |
| DHAMAIL_007 | দেইখে আইলাম তারে সইগো | Dhamail |
| DHAMAIL_009 | ধামাইল শেষ হইয়া যায় | Dhamail |
| DHAMAIL_010 | আমরা কার বাড়িত আইলাম গো | Dhamail |
| GARBA_018 | Dholida Dhol Re Vagad | Garba |
| GHOOMAR_016 | Jeero | Ghoomar |
| JHUMUR_009 | Moi to Jabo Kam Jari | Jhumur |
| LAVANI_033 | Dolyavar Gogal Lava | Lavani |

### suddha kalyan
| Song ID | Song | Genre |
|---|---|---|
| BHANGRA_030 | Sucha Soorma (Balbiro Bhabhi) | Bhangra |
| BIHU_001 | Morom Logai Logai | Bihu |
| BIHU_003 | Kokalote Tongali | Bihu |
| BIHU_011 | Eijon Korenu Nasoni | Bihu |
| BIHU_031 | Kahili puwate | Bihu |
| DHAMAIL_026 | জলার জলার ঘাটে | Dhamail |
| GARBA_004 | Meldimaa Na Kumkum Pagla | Garba |
| GARBA_015 | Hindole Zule Amba | Garba |
| GHOOMAR_003 | Choumaso | Ghoomar |
| GHOOMAR_009 | Gorband | Ghoomar |
| GHOOMAR_019 | Run Jhun Baje Ghooghara | Ghoomar |
| GHOOMAR_021 | Dheere Chhal Re Panihari | Ghoomar |
| JHUMUR_0011 | Barire Simare | Jhumur |
| LAVANI_006 | Kalidar Kapuri Paan | Lavani |
| LAVANI_009 | Chabidar Chabi | Lavani |
| LAVANI_020 | Themba Thembana Kelai | Lavani |
| LAVANI_024 | Majha Taat Kar Aadhi | Lavani |
| MAAND_020 | Rajasthani Folk Raag Maand | Maand |
| MAAND_021 | Chaudhary | Maand |
| MAAND_022 | Chhap Tilak Sab Cheeni | Maand |

## PART 7 — STRONGEST RETRIEVALS

### Top 25 Highest-Margin Recordings

- **BAUL_023** (Baul): Manush Bhajo Manusheyi Pujo by Purna Das Baul | **Top-1:** Purba (0.9455) | **Top-2:** N/A (0.0000) | **Margin:** 0.9455
- **GARBA_023** (Garba): Jay Aadhyashakti by Hemant Chauhan | **Top-1:** Purba (0.9348) | **Top-2:** N/A (0.0000) | **Margin:** 0.9348
- **GHOOMAR_019** (Ghoomar): Run Jhun Baje Ghooghara by Seema Mishra | **Top-1:** Alhaiya Bilaval (0.9360) | **Top-2:** suddha kalyan (0.9205) | **Margin:** 0.0156
- **BHANGRA_007** (Bhangra): Tunak Tunak Tun by Daler Mehndi | **Top-1:** Megh (0.9487) | **Top-2:** Deepak (0.9359) | **Margin:** 0.0127
- **DHAMAIL_021** (Dhamail): জলের ঘাটে by Pausali Banerjee | **Top-1:** Bihag (0.9414) | **Top-2:** Yaman Kalyan (0.9299) | **Margin:** 0.0115
- **JHUMUR_0012** (Jhumur): Mor Dhoni Go by Zubeen Garg | **Top-1:** Malkauns (0.9293) | **Top-2:** Madhuvanti (0.9192) | **Margin:** 0.0101
- **BHANGRA_031** (Bhangra): Nawan Le Liya Truck Tere Yaar Ne (Truck Billiya) by Surinder Shinda | **Top-1:** Rageshree (0.9379) | **Top-2:** Khamaj (0.9289) | **Margin:** 0.0090
- **BAUL_024** (Baul): Dil Dariyar Majhete by Purna Das Baul | **Top-1:** Purba (0.9334) | **Top-2:** Bageshree (0.9246) | **Margin:** 0.0088
- **LAVANI_017** (Lavani): Kashyala Lavato by Bela Shende | **Top-1:** Barwa (0.9223) | **Top-2:** Darbari (0.9137) | **Margin:** 0.0086
- **LAVANI_012** (Lavani): Naki Doli Chhaan by Usha Mangeshkar | **Top-1:** Puriya (0.9578) | **Top-2:** Kedar (0.9494) | **Margin:** 0.0084
- **BAUL_021** (Baul): Prem Koraa Ki Jwalaa by Purna Das Baul | **Top-1:** Purba (0.9417) | **Top-2:** Gauri (0.9334) | **Margin:** 0.0084
- **BIHU_019** (Bihu): O Dhei Oi by Khagen Mahanta | **Top-1:** Megh (0.9416) | **Top-2:** Bhairavi (0.9335) | **Margin:** 0.0081
- **BHANGRA_024** (Bhangra): Apna Sangeet Vajeh Apna Sangeet by Apna Sangeet | **Top-1:** Bilaval (0.9226) | **Top-2:** Desh (0.9145) | **Margin:** 0.0081
- **BHANGRA_014** (Bhangra): Ambran Tu Ayi Hui Hoor Sohniye by Balwinder Safri | **Top-1:** Gujari Todi (0.9384) | **Top-2:** Megh (0.9304) | **Margin:** 0.0081
- **MAAND_010** (Maand): Badila Yahi Rahiyo Saa by Banarsi Babu | **Top-1:** Gauri (0.9537) | **Top-2:** Bageshree (0.9457) | **Margin:** 0.0080
- **BAUL_011** (Baul): Chor Poreche Babur Bagane by Purna Das Baul | **Top-1:** Bhairavi (0.9281) | **Top-2:** Megh (0.9205) | **Margin:** 0.0076
- **SAMBALPURI_006** (Sambalpuri): Rasia by Mantu Chhuria | **Top-1:** Bilaval (0.9351) | **Top-2:** Bageshree (0.9276) | **Margin:** 0.0074
- **BHANGRA_009** (Bhangra): Ho Jayegi Balle Balle by Daler Mehndi | **Top-1:** Darbari (0.9468) | **Top-2:** Basant (0.9397) | **Margin:** 0.0071
- **DHAMAIL_015** (Dhamail): চান্নি রাইতে আইও কালা by Habib Simu/Shimu | **Top-1:** Madhuvanti (0.9452) | **Top-2:** Miyan Ki Todi orTodi or Darbari Todi (0.9382) | **Margin:** 0.0070
- **SAMBALPURI_028** (Sambalpuri): Kulfirani Chocobar by Ira Mohanty | **Top-1:** Marva (0.9496) | **Top-2:** Megh (0.9429) | **Margin:** 0.0067
- **BHANGRA_026** (Bhangra): Kuriyan Ne Paya Bhangra by Kuldeep Manak | **Top-1:** Darbari (0.9309) | **Top-2:** Asavari (0.9244) | **Margin:** 0.0065
- **GHOOMAR_002** (Ghoomar): Gangore/Gangaur by Seema Mishra | **Top-1:** Bhairav (0.9523) | **Top-2:** Miyan Ki Todi orTodi or Darbari Todi (0.9458) | **Margin:** 0.0064
- **JHUMUR_004** (Jhumur): Sajanigo by Dulal Manki | **Top-1:** Desi (0.9391) | **Top-2:** Yaman or Kalyan (0.9328) | **Margin:** 0.0063
- **BHANGRA_021** (Bhangra): Kinni Sohni by Malkit Singh (Golden Star) | **Top-1:** Darbari (0.9421) | **Top-2:** Alhaiya Bilaval (0.9358) | **Margin:** 0.0063
- **GHOOMAR_020** (Ghoomar): Kangasiyo by Seema Mishra | **Top-1:** Ahir Bhairav (0.9296) | **Top-2:** Desh (0.9234) | **Margin:** 0.0062

## PART 8 — MOST AMBIGUOUS RETRIEVALS

### Top 25 Lowest-Margin Recordings
Low margin means the retrieval system found competing Raag candidates with relatively similar similarity scores.

- **BIHU_012** (Bihu): Laje Moru Moru by Bhupen Hazarika | **Top-1:** Bhoopali (0.9350) | **Top-2:** Puriya (0.9348) | **Margin:** 0.0002
- **BAUL_008** (Baul): Tripti by Purna Das Baul | **Top-1:** Malkauns (0.8812) | **Top-2:** Sorath (0.8810) | **Margin:** 0.0002
- **MAAND_035** (Maand): Rajwadi Mehfil by Nand Lal Bhat | **Top-1:** Alhaiya Bilaval (0.9365) | **Top-2:** Marva (0.9363) | **Margin:** 0.0002
- **BHANGRA_018** (Bhangra): Tutak Tutak Tutiya by Malkit Singh (Golden Star) | **Top-1:** Darbari (0.9297) | **Top-2:** Alhaiya Bilaval (0.9295) | **Margin:** 0.0002
- **DHAMAIL_026** (Dhamail): জলার জলার ঘাটে by Himel Dev Rahul | **Top-1:** Marva (0.9157) | **Top-2:** Ahir Bhairav (0.9156) | **Margin:** 0.0002
- **BHANGRA_029** (Bhangra): Putt Jattan De by Surinder Shinda | **Top-1:** Abhogi (0.9170) | **Top-2:** Darbari (0.9168) | **Margin:** 0.0002
- **SAMBALPURI_008** (Sambalpuri): Hae Go by Mantu Chhuria | **Top-1:** Darbari (0.9347) | **Top-2:** Megh (0.9345) | **Margin:** 0.0002
- **JHUMUR_0019** (Jhumur): Sonamoni by Amrit Tanti | **Top-1:** Adana (0.9368) | **Top-2:** Desh (0.9367) | **Margin:** 0.0002
- **LAVANI_030** (Lavani): Ghotala Ghotala Zala by Vaishali Samant | **Top-1:** Bilaskhani Todi (0.9362) | **Top-2:** Marva (0.9360) | **Margin:** 0.0001
- **BHANGRA_001** (Bhangra): Tera Yaar Bolda by Surjit Bindrakhia | **Top-1:** Bahar (0.9212) | **Top-2:** Darbari (0.9210) | **Margin:** 0.0001
- **DHAMAIL_016** (Dhamail): যুগল মিলন হইলো গো by Jumiar Shanta | **Top-1:** Bihag (0.9409) | **Top-2:** Kedar (0.9408) | **Margin:** 0.0001
- **SAMBALPURI_013** (Sambalpuri): Ramagaru by Asima Panda | **Top-1:** Sorath (0.9276) | **Top-2:** Desh (0.9275) | **Margin:** 0.0001
- **BIHU_013** (Bihu): Ki Sai Aso by Bhupen Hazarika | **Top-1:** Ahir Bhairav (0.9378) | **Top-2:** Asavari (0.9377) | **Margin:** 0.0001
- **LAVANI_026** (Lavani): Bai Mazi Karangali Modali by Asha Bhosle | **Top-1:** Basant Mukhari (0.9551) | **Top-2:** Puriya (0.9550) | **Margin:** 0.0001
- **SAMBALPURI_033** (Sambalpuri): Din Gani Gani by Jitendra Haripal | **Top-1:** Shree (0.9340) | **Top-2:** Alhaiya Bilaval (0.9339) | **Margin:** 0.0001
- **BAUL_019** (Baul): Aami Ek Pagal Baul by Purna Das Baul | **Top-1:** Bhairavi (0.9208) | **Top-2:** Ahir Bhairav (0.9207) | **Margin:** 0.0001
- **MAAND_022** (Maand): Chhap Tilak Sab Cheeni by Kirti Sagathia | **Top-1:** Adana (0.9323) | **Top-2:** Bahar (0.9323) | **Margin:** 0.0001
- **MAAND_019** (Maand): Raag Maand Bandish by Mame Khan | **Top-1:** Bilaval (0.9388) | **Top-2:** Purba (0.9388) | **Margin:** 0.0000
- **BHANGRA_003** (Bhangra): Dupatta Tera Satrang Da by Surjit Bindrakhia | **Top-1:** Darbari (0.9340) | **Top-2:** Alhaiya Bilaval (0.9340) | **Margin:** 0.0000
- **JHUMUR_0023** (Jhumur): Koliye Juge Alai Doyal by Navin Tanti | **Top-1:** Lalit (0.9528) | **Top-2:** Megh (0.9527) | **Margin:** 0.0000
- **JHUMUR_0024** (Jhumur): Laal Pari by Bejnath Tanti | **Top-1:** Madhuvanti (0.9358) | **Top-2:** Sorath (0.9358) | **Margin:** 0.0000
- **JHUMUR_002** (Jhumur): Nistei by Dulal Manki | **Top-1:** Bageshree (0.9418) | **Top-2:** Sohni (0.9418) | **Margin:** 0.0000
- **BIHU_004** (Bihu): Tumake Logot by Zubeen Garg | **Top-1:** Bilaval (0.9349) | **Top-2:** Puriya (0.9349) | **Margin:** 0.0000
- **JHUMUR_0013** (Jhumur): Jhuli Jhule by Zubeen Garg | **Top-1:** Desi (0.9251) | **Top-2:** Madhuvanti (0.9251) | **Margin:** 0.0000
- **BIHU_021** (Bihu): Petore Juruni Bhat by Khagen Mahanta | **Top-1:** Megh Malhar (0.9354) | **Top-2:** Malhar (0.9354) | **Margin:** 0.0000

## PART 9 — IMPORTANT SCIENTIFIC WARNING

The Folk recordings do not have verified Raag ground-truth labels.
Therefore:
**Predicted Raag = candidate retrieved from TRF reference space.**
It does NOT equal verified Raag identity.
The SARAGA cross-dataset evaluation should be used to contextualize the reliability of this retrieval approach.

## PART 10 — FINAL SUMMARY

- 328 total canonical recordings
- 327 classified
- 1 excluded
- 2310 total 60-second sequences
- 53 unique retrieved Raags
- Top 10 retrieved Raags: Darbari, Adana, Marva, Khamaj, Bahar, Ahir Bhairav, Yaman Kalyan, Bhoopali, Bilaval, Asavari
- Mean similarity margin: 0.0082
- Median similarity margin: 0.0019
- Most concentrated Folk genres: Bhangra, Ghoomar, Dhamail
- Most frequently retrieved Raags: Darbari, Adana, Marva
- Any technical anomalies: None found. All 328 canonical recordings accurately accounted for.
