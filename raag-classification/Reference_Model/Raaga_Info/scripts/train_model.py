import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from dataset import SaragaDataset
from model import RaagaCNN

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    dataset = SaragaDataset()

    num_classes = len(dataset.encoder.classes_)

    print(f"\nNumber of Raagas : {num_classes}")
    print(f"Total Samples    : {len(dataset)}")

    # ======================================================
    # Train / Validation Split
    # ======================================================

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"\nTraining Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}")

    # ======================================================
    # Data Loaders
    # ======================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ======================================================
    # Model
    # ======================================================

    model = RaagaCNN(num_classes)

    model.to(DEVICE)

    # ======================================================
    # Loss
    # ======================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    print("\nModel Ready!")
    print("=" * 60)

    return (
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer
    )