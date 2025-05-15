import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import random
import stempeg
from einops import rearrange
from datasets import load_dataset
from model import ConvUNet
from tqdm import tqdm
from torch.serialization import add_safe_globals
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio.transforms as T

# Add numpy's _reconstruct to safe globals for loading
add_safe_globals(['numpy.core.multiarray._reconstruct'])

# call functions to get transformed audio
class StemsDataset(Dataset):
    def __init__(self, file_list, processed_dir='processed_stems'):
        self.file_list = file_list
        self.processed_dir = processed_dir
        # Add data augmentation transforms
        self.time_stretch = T.TimeStretch()
        self.pitch_shift = T.PitchShift(sample_rate=44100)

    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        base_name = os.path.splitext(os.path.basename(filename))[0]
        processed_path = os.path.join(self.processed_dir, base_name + '.pt')
        
        if not os.path.exists(processed_path):
            raise ValueError(f"Processed file not found: {processed_path}")
            
        # Load with weights_only=False to allow numpy arrays
        processed_stems = torch.load(processed_path, weights_only=False)

        # Get mix magnitude and add channel dimension
        mix_mag = torch.tensor(processed_stems['mix']['magnitude'], dtype=torch.float32)
        
        # Pad or crop to nearest power of 2 for both dimensions
        target_time = 512  # This should be a power of 2
        target_freq = 1024  # This should be a power of 2
        current_time = mix_mag.shape[1]
        current_freq = mix_mag.shape[0]
        
        # Handle time dimension
        if current_time > target_time:
            # Crop from the middle
            start_time = (current_time - target_time) // 2
            mix_mag = mix_mag[:, start_time:start_time + target_time]
        elif current_time < target_time:
            # Pad with zeros
            pad_time = target_time - current_time
            mix_mag = torch.nn.functional.pad(mix_mag, (0, pad_time))
            
        # Handle frequency dimension
        if current_freq > target_freq:
            # Crop from the middle
            start_freq = (current_freq - target_freq) // 2
            mix_mag = mix_mag[start_freq:start_freq + target_freq, :]
        elif current_freq < target_freq:
            # Pad with zeros
            pad_freq = target_freq - current_freq
            mix_mag = torch.nn.functional.pad(mix_mag, (0, 0, 0, pad_freq))
            
        mix_mag = mix_mag.unsqueeze(0)  # Add channel dimension: [freq_bins, time_frames] -> [1, freq_bins, time_frames]

        # Process target stems
        target_stems = ['drums', 'bass', 'other', 'vocals']
        targets = []
        for stem in target_stems:
            target = torch.tensor(processed_stems[stem]['magnitude'], dtype=torch.float32)
            
            # Apply the same padding/cropping to targets
            if current_time > target_time:
                target = target[:, start_time:start_time + target_time]
            elif current_time < target_time:
                target = torch.nn.functional.pad(target, (0, pad_time))
                
            if current_freq > target_freq:
                target = target[start_freq:start_freq + target_freq, :]
            elif current_freq < target_freq:
                target = torch.nn.functional.pad(target, (0, 0, 0, pad_freq))
                
            targets.append(target)
        
        target = torch.stack(targets, dim=0)  # Stack to get [4, freq_bins, time_frames]
        
        # Apply data augmentation during training
        if self.training:
            # Random time stretch
            if random.random() < 0.5:
                stretch_factor = random.uniform(0.9, 1.1)
                mix_mag = self.time_stretch(mix_mag, stretch_factor)
                target = self.time_stretch(target, stretch_factor)
            
            # Random pitch shift
            if random.random() < 0.5:
                pitch_shift = random.randint(-2, 2)
                mix_mag = self.pitch_shift(mix_mag, pitch_shift)
                target = self.pitch_shift(target, pitch_shift)
        
        return mix_mag, target


# Get the list of preprocessed files
processed_dir = 'processed_stems'
train_dir = os.path.join(processed_dir, 'train')
test_dir = os.path.join(processed_dir, 'test')

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all preprocessed files
all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
print(f"Found {len(all_files)} preprocessed files")

# Split into train and test (80/20 split)
random.seed(42)  # For reproducibility
random.shuffle(all_files)
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

print(f"Split into {len(train_files)} training files and {len(test_files)} test files")

# Create full paths
train_file_list = [os.path.join(processed_dir, f) for f in train_files]
test_file_list = [os.path.join(processed_dir, f) for f in test_files]

train_dataset = StemsDataset(file_list=train_file_list)
test_dataset = StemsDataset(file_list=test_file_list)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    learning_rate = 0.001  # Increased initial learning rate
    epochs = 100  # Increased number of epochs
    weight_decay = 1e-4

    model = ConvUNet(out_channels=4)  # 4 channels for 4 stems
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Combined loss function
    l1_criterion = torch.nn.L1Loss()
    mse_criterion = torch.nn.MSELoss()
    
    def combined_loss(pred, target):
        l1_loss = l1_criterion(pred, target)
        mse_loss = mse_criterion(pred, target)
        return l1_loss + 0.5 * mse_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Gradient clipping
    max_grad_norm = 1.0

    best_test_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = combined_loss(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Validation phase
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Validation")):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = combined_loss(output, target)
                test_loss += loss.item()

        test_loss = test_loss / len(test_loader)
        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}')
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_song_splitter_model.pth')
            print(f"Saved new best model with test loss: {test_loss:.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'song_splitter_model.pth')

if __name__ == '__main__':
    main()