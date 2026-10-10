import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os
import time

from swaragram_dataset import SwaragramDataset
from model import RaagaCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_swaragram_model(in_memory=False):
    print("=" * 60)
    print("Loading Swaragram Dataset...")
    print("=" * 60)

    dataset = SwaragramDataset(in_memory=in_memory)
    num_classes = len(dataset.encoder.classes_)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use deterministic split generator
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"\nTraining Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")

    # If in_memory=True, passing a 2.5GB python object to spawn workers is highly inefficient/fatal.
    # Therefore, when using RAM cache, we strictly use num_workers=0.
    workers = 0 if in_memory else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # Optimized batch size
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if workers > 0 else None
    )

    model = RaagaCNN(num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    print("\nModel Ready!")
    print("=" * 60)
    return model, train_loader, val_loader, criterion, optimizer, scheduler


def main():
    set_seed(42)
    
    IN_MEMORY_CACHE = True  # Set to True for RAM cache mode
    
    model, train_loader, val_loader, criterion, optimizer, scheduler = prepare_swaragram_model(in_memory=IN_MEMORY_CACHE)
    EPOCHS = 20
    start_epoch = 0
    best_accuracy = 0.0
    
    checkpoint_path = "latest_checkpoint_swaragram.pth"
    best_model_path = "best_model_swaragram.pth"
    
    # Check for resumable checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print(f"Resumed from Epoch {start_epoch} with Best Accuracy: {best_accuracy:.2f}%")
        
    epoch_times = []

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Current LR: {current_lr:.6f}")
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for swara, labels in train_loader:
            swara = swara.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(swara)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for swara, labels in val_loader:
                swara = swara.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(swara)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        scheduler.step(val_accuracy)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        avg_time_per_epoch = sum(epoch_times) / len(epoch_times)
        remaining_epochs = EPOCHS - (epoch + 1)
        est_remaining_time = avg_time_per_epoch * remaining_epochs
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"Epoch Duration: {epoch_duration:.2f}s | Est. Remaining: {est_remaining_time/60:.2f}m")

        # Save latest checkpoint for resumption
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': best_accuracy
        }
        torch.save(checkpoint_dict, checkpoint_path)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(checkpoint_dict, best_model_path)
            print(f"*** New Best Model Saved! (Acc: {best_accuracy:.2f}%) ***")

    print("\nTraining Finished!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()
