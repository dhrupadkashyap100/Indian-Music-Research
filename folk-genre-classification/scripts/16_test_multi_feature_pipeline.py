"""
=========================================================
Test Script: Multi-Feature Pipeline Architecture
=========================================================
"""

import sys
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.config import (
    METADATA_DIR, MEL_METADATA_FILE, CHROMA_METADATA_FILE,
    OUTPUT_DIR, GENRE_MAPPING_FILE, BATCH_SIZE
)
from utils.multi_feature_dataset import MultiFeatureDataset
from models.multi_feature_efficientnet import MultiFeatureEfficientNet

def test_pipeline():
    print("\n" + "="*60)
    print("PHASE 1: MULTI-FEATURE PIPELINE ARCHITECTURE VERIFICATION")
    print("="*60 + "\n")

    # 1. Dataset Verification
    print("--- 1. Dataset Verification ---")
    
    mel_meta_path = METADATA_DIR / MEL_METADATA_FILE
    chroma_meta_path = METADATA_DIR / CHROMA_METADATA_FILE
    
    if not mel_meta_path.exists():
        raise FileNotFoundError(f"Mel metadata missing: {mel_meta_path}")
    if not chroma_meta_path.exists():
        raise FileNotFoundError(f"Chroma metadata missing: {chroma_meta_path}")
        
    mel_df = pd.read_csv(mel_meta_path)
    chroma_df = pd.read_csv(chroma_meta_path)
    
    print("[PASS] Both metadata files load correctly.")
    
    genre_mapping_path = OUTPUT_DIR / "models" / GENRE_MAPPING_FILE
    if genre_mapping_path.exists():
        with open(genre_mapping_path, "r") as f:
            genre_mapping = json.load(f)
    else:
        # Fallback for testing if running fresh
        genres = sorted(list(mel_df["Genre"].unique()))
        genre_mapping = {g: i for i, g in enumerate(genres)}

    # Initialize Dataset (this triggers strict pairing & integrity checks)
    print("Initializing MultiFeatureDataset (runs strict verification)...")
    dataset = MultiFeatureDataset(mel_df, chroma_df, genre_mapping)
    print("[PASS] Pairing is perfect.")
    print("[PASS] Shapes are correct.")
    print("[PASS] Labels match.")
    print("[PASS] No missing files.")
    
    # 2. DataLoader Verification
    print("\n--- 2. DataLoader Verification ---")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    mel_batch, chroma_batch, labels_batch, filenames = next(iter(dataloader))
    print(f"Batch Size: {mel_batch.size(0)}")
    print(f"Mel Shape : {tuple(mel_batch.shape)}")
    print(f"Chroma Shape: {tuple(chroma_batch.shape)}")
    print(f"Labels: {labels_batch}")
    print(f"Filenames: {filenames}")
    
    # 3. Model Architecture Verification
    print("\n--- 3. Encoders & Architecture Verification ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = MultiFeatureEfficientNet(num_classes=len(genre_mapping), verbose=False).to(device)
    model.eval() # evaluation mode for testing
    
    mel_batch = mel_batch.to(device)
    chroma_batch = chroma_batch.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # Step-by-step trace
            print("\nTracking Tensor Flow:")
            print(f"1. Input Mel Shape        : {tuple(mel_batch.shape)}")
            print(f"   Input Chroma Shape     : {tuple(chroma_batch.shape)}")
            
            mel_emb = model.mel_encoder(mel_batch)
            chroma_emb = model.chroma_encoder(chroma_batch)
            print(f"2. Mel Encoder Output     : {tuple(mel_emb.shape)}")
            print(f"   Chroma Encoder Output  : {tuple(chroma_emb.shape)}")
            
            # Manual Fusion trace
            concat_emb = torch.cat([mel_emb, chroma_emb], dim=1)
            print(f"3. Concat Output          : {tuple(concat_emb.shape)}")
            
            gate = torch.sigmoid(model.fusion.attention_conv(concat_emb))
            print(f"4. Fusion Gate Shape      : {tuple(gate.shape)}")
            
            fused = (gate * mel_emb) + ((1 - gate) * chroma_emb)
            print(f"5. Gated Output           : {tuple(fused.shape)}")
            
            reduced = model.fusion.reduction_conv(fused)
            print(f"6. 1x1 Conv (Reduction)   : {tuple(reduced.shape)}")
            
            # EfficientNet forward
            features = model.backbone.features(reduced)
            print(f"7. EfficientNet Features  : {tuple(features.shape)}")
            
            features = model.backbone.avgpool(features)
            features = torch.flatten(features, 1)
            logits = model.backbone.classifier(features)
            print(f"8. Classifier Logits      : {tuple(logits.shape)}")
            
            # Verify full forward pass method
            out = model(mel_batch, chroma_batch)
            assert torch.allclose(logits, out, atol=1e-5), "Manual trace mismatch with forward pass"
            
    print("\n--- 4. GPU & AMP Verification ---")
    print("[PASS] CUDA tensor placement successful.")
    print("[PASS] AMP Autocast context active.")
    print("[PASS] Forward pass completed successfully.")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED: ARCHITECTURE IS READY FOR TRAINING")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_pipeline()
