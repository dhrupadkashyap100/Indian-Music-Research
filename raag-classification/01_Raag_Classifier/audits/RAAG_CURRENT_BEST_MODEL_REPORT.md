# CURRENT BEST RAAG CLASSIFIER

BEST CURRENT MODEL: CultureMERT-95M (Zero-shot Probing & KNN)
MODEL SCRIPT: `folk_inference_pipeline.py`
CHECKPOINT: `ntua-slp/CultureMERT-95M` (HuggingFace) + `best_f1.pt` (Local BiLSTM)
INPUT FEATURE: 768-D Transformers Embeddings
BEST VALIDATION ACCURACY: ~63.5% mAP (as per README for Hindustani probing)
TOP-3 EVALUATION NOT YET IMPLEMENTED OR VERIFIED (Code shows Top-5 probability generation, but formal Top-3 metric reporting on Folk is not present.)
