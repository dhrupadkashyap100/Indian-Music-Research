import torch
import sys
import os
from pathlib import Path

# Add scripts directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from train_swaragram import prepare_swaragram_model, DEVICE
import time

def verify_pipeline():
    print("Phase 1: Instantiating Dataset and DataLoaders via prepare_swaragram_model()...")
    start_time = time.time()
    
    # This inherently tests Phase 2, 3, and part of 4
    model, train_loader, val_loader, criterion, optimizer = prepare_swaragram_model()
    
    print("\nPhase 2 & 3: DataLoader Verification")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get one batch
    chroma_batch, label_batch = next(iter(train_loader))
    print(f"Batch shape: {chroma_batch.shape}, dtype: {chroma_batch.dtype}")
    print(f"Label shape: {label_batch.shape}, dtype: {label_batch.dtype}")
    
    assert chroma_batch.shape == (32, 1, 12, 938), f"Unexpected shape {chroma_batch.shape}"
    
    print("\nPhase 4 & 5: Dry Training Run (2 batches)")
    model.train()
    
    for i, (swara, labels) in enumerate(train_loader):
        if i >= 2:
            break
            
        print(f"\n--- Batch {i+1} ---")
        swara = swara.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(swara)
        print(f"Output shape: {outputs.shape}")
        
        # Loss
        loss = criterion(outputs, labels)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        print(f"Gradients computed: {has_grad}")
        
        # Step
        optimizer.step()
        print("Optimizer stepped.")
        
    print("\nPhase 6: Checkpoint Generation")
    test_ckpt_path = "test_best_model_swaragram.pth"
    torch.save(model.state_dict(), test_ckpt_path)
    if os.path.exists(test_ckpt_path):
        print(f"Checkpoint successfully saved to {test_ckpt_path}")
        os.remove(test_ckpt_path)
    else:
        print("Failed to save checkpoint!")
        
    end_time = time.time()
    print(f"\nVerification Complete in {end_time - start_time:.2f} seconds.")
    print("✅ Ready for Training")

if __name__ == "__main__":
    verify_pipeline()
