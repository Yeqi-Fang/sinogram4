import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class SinogramDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None, test=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        # Determine dataset range based on train/test
        if not test:
            if is_train:
                self.i_range = range(1, 171)  # 1 to 170 (placeholder for demo)
            else:
                self.i_range = range(1, 37)   # 1 to 37 (placeholder for demo)
        else:
            if is_train:
                self.i_range = range(1, 171 - 169)  # 1 to 170
            else:
                self.i_range = range(1, 37 - 35)   # 1 to 36
                
        self.j_range = range(1, 1765)  # 1 to 1764
        
        # Create all possible (i,j) pairs
        self.pairs = [(i, j) for i in self.i_range for j in self.j_range]
        
        print(f"Dataset initialized with {len(self.pairs)} pairs of sinograms")
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_file(self, i, j, is_complete=False):
        """Helper method to load a single file from disk"""
        prefix = "complete" if is_complete else "incomplete"
        file_path = os.path.join(self.data_dir, f"{prefix}_{i}_{j}.npy")
        return np.load(file_path).astype(np.float32)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # Determine position in group (0-indexed): 0 means first image in a group, 41 means last image in a group
        pos_in_group = (j - 1) % 42
        max_j = max(self.j_range)

        # --- For the incomplete sinogram ---
        # Load current sinogram and ensure channel dimension exists
        current_incomplete = torch.from_numpy(self._load_file(i, j, False))
        if current_incomplete.dim() == 2:
            current_incomplete = current_incomplete.unsqueeze(0)
        
        # Left neighbor: if first in group, use current image itself; otherwise load j-1
        if pos_in_group == 0:
            left_incomplete = current_incomplete.clone()
        else:
            left_incomplete = torch.from_numpy(self._load_file(i, j - 1, False))
            if left_incomplete.dim() == 2:
                left_incomplete = left_incomplete.unsqueeze(0)
        
        # Right neighbor: if last in group, use current image itself; otherwise load j+1
        if pos_in_group == 41:
            right_incomplete = current_incomplete.clone()
        else:
            right_incomplete = torch.from_numpy(self._load_file(i, j + 1, False))
            if right_incomplete.dim() == 2:
                right_incomplete = right_incomplete.unsqueeze(0)
        
        # Previous cycle (j-42): if j-42 < 1, use current image
        if j < 43:  # j-42 would be < 1, which is out of range
            prev_cycle_incomplete = current_incomplete.clone()
        else:
            prev_cycle_incomplete = torch.from_numpy(self._load_file(i, j - 42, False))
            if prev_cycle_incomplete.dim() == 2:
                prev_cycle_incomplete = prev_cycle_incomplete.unsqueeze(0)
        
        # Next cycle (j+42): if j+42 > max_j, use current image
        if j > max_j - 42:  # j+42 would be > max_j, which is out of range
            next_cycle_incomplete = current_incomplete.clone()
        else:
            next_cycle_incomplete = torch.from_numpy(self._load_file(i, j + 42, False))
            if next_cycle_incomplete.dim() == 2:
                next_cycle_incomplete = next_cycle_incomplete.unsqueeze(0)
        
        # Stack to create a 5-channel tensor (5, H, W) with order: prev_cycle, left, current, right, next_cycle
        incomplete_5ch = torch.cat([prev_cycle_incomplete, left_incomplete, current_incomplete, 
                                  right_incomplete, next_cycle_incomplete], dim=0)
        
        # --- For the complete sinogram ---
        current_complete = torch.from_numpy(self._load_file(i, j, True))
        if current_complete.dim() == 2:
            current_complete = current_complete.unsqueeze(0)
        
        if pos_in_group == 0:
            left_complete = current_complete.clone()
        else:
            left_complete = torch.from_numpy(self._load_file(i, j - 1, True))
            if left_complete.dim() == 2:
                left_complete = left_complete.unsqueeze(0)
        
        if pos_in_group == 41:
            right_complete = current_complete.clone()
        else:
            right_complete = torch.from_numpy(self._load_file(i, j + 1, True))
            if right_complete.dim() == 2:
                right_complete = right_complete.unsqueeze(0)
        
        # Previous cycle (j-42): if j-42 < 1, use current image
        if j < 43:
            prev_cycle_complete = current_complete.clone()
        else:
            prev_cycle_complete = torch.from_numpy(self._load_file(i, j - 42, True))
            if prev_cycle_complete.dim() == 2:
                prev_cycle_complete = prev_cycle_complete.unsqueeze(0)
        
        # Next cycle (j+42): if j+42 > max_j, use current image
        if j > max_j - 42:
            next_cycle_complete = current_complete.clone()
        else:
            next_cycle_complete = torch.from_numpy(self._load_file(i, j + 42, True))
            if next_cycle_complete.dim() == 2:
                next_cycle_complete = next_cycle_complete.unsqueeze(0)
        
        # Stack to create a 5-channel tensor (5, H, W) with order: prev_cycle, left, current, right, next_cycle
        complete_5ch = torch.cat([prev_cycle_complete, left_complete, current_complete, 
                                right_complete, next_cycle_complete], dim=0)
        
        # Apply transforms if provided
        if self.transform:
            incomplete_5ch = self.transform(incomplete_5ch)
            complete_5ch = self.transform(complete_5ch)
        
        return incomplete_5ch, complete_5ch

# Example of how to use the dataset
def create_dataloaders(data_dir, batch_size=8, num_workers=4, test=False, transform=False):
    # Define transforms
    if transform:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the specified dimensions
        ])
    else:
        transform = None
    
    # Create datasets
    train_dataset = SinogramDataset(os.path.join(data_dir, 'train'), is_train=True, transform=transform, test=test)
    test_dataset = SinogramDataset(os.path.join(data_dir, 'test'), is_train=False, transform=transform, test=test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader