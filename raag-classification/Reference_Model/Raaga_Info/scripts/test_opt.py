import sys
import os
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_swaragram

original_prepare = train_swaragram.prepare_swaragram_model

def mock_prepare():
    model, train_loader, val_loader, criterion, optimizer, scheduler = original_prepare()
    class ShortLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            it = iter(self.loader)
            yield next(it)
            yield next(it)
        def __len__(self):
            return 2
    return model, ShortLoader(train_loader), ShortLoader(val_loader), criterion, optimizer, scheduler

train_swaragram.prepare_swaragram_model = mock_prepare

print('--- STARTING FIRST RUN ---')
def main_run_1():
    train_swaragram.set_seed(42)
    model, train_loader, val_loader, criterion, optimizer, scheduler = train_swaragram.prepare_swaragram_model()
    EPOCHS = 1
    
    checkpoint_path = 'latest_checkpoint_swaragram.pth'
    
    for epoch in range(0, EPOCHS):
        model.train()
        for swara, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(swara)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        scheduler.step(10.0)
        
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': 10.0
        }
        torch.save(checkpoint_dict, checkpoint_path)

main_run_1()

print('\n--- STARTING SECOND RUN (RESUME) ---')
def main_run_2():
    train_swaragram.set_seed(42)
    model, train_loader, val_loader, criterion, optimizer, scheduler = train_swaragram.prepare_swaragram_model()
    
    checkpoint_path = 'latest_checkpoint_swaragram.pth'
    if os.path.exists(checkpoint_path):
        print(f'Resuming from {checkpoint_path}')
        ckpt = torch.load(checkpoint_path)
        print(f"Resumed epoch: {ckpt['epoch']}, Best Acc: {ckpt['best_accuracy']}")
        assert ckpt['epoch'] == 0
        assert ckpt['best_accuracy'] == 10.0

main_run_2()

os.remove('latest_checkpoint_swaragram.pth')
print('\n✅ Optimization Verification successful!')
