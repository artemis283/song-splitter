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

# cut audio to 30 seconds
def preprocess_audio(filename, time, fft_size=2048, hop_length=512, save_dir='processed_stems'):
    # read audio
    if not filename.endswith('.stem.mp4'):
        return None
        
    audio, rate = stempeg.read_stems(filename, stem_id=None, sample_rate=44100)
    num_samples = time * rate

    processed_stems = {}

    for name, stem in zip(['mix', 'drums', 'bass', 'other', 'vocals'], audio): 
        if stem.shape[0] > num_samples:
            stem = stem[:num_samples]
        else:
            stem = np.pad(stem, ((0, num_samples - stem.shape[0]), (0, 0)), mode='constant')
        stem = stem.mean(axis=1)

        # maybe deal with multiple 30 second clips for each song

        audio = stem.reshape(1, -1)
        stft = librosa.stft(audio[0], n_fft=fft_size, hop_length=hop_length)
        magnitude = np.abs(stft)
        processed_stems[name] = {
            'audio': audio,
            'stft': stft,
            'magnitude': magnitude,
        }
    
    return processed_stems


# call functions to get transformed audio
class StemsDataset(Dataset):
    def __init__(self, file_list, time=30, fft_size=2048, hop_length=512):
        self.file_list = file_list
        self.time = time
        self.fft_size = fft_size
        self.hop_length = hop_length

    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        processed_stems = preprocess_audio(filename, self.time, self.fft_size, self.hop_length)
        
        if processed_stems is None:
            raise ValueError(f"Failed to process {filename}")

        mix_mag = torch.tensor(processed_stems['mix']['magnitude'], dtype=torch.float32).unsqueeze(1) # unsqueeze to add channel dimension 

        target_stems = ['drums', 'bass', 'other', 'vocals']
        targets = [torch.tensor(processed_stems[stem]['magnitude'], dtype=torch.float32) for stem in target_stems]
        target = torch.stack(targets, dim=0)
        
        return mix_mag, target


# Get list of files
train_dir = '/Users/artemiswebster/source/song-splitter/musdb18/train'
file_list = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.stem.mp4')]
print(f"Found {len(file_list)} stem files")

train_dataset = StemsDataset(file_list=file_list)
test_dataset = StemsDataset(file_list=file_list)  # For now, using same files for test

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get one batch to see shapes
for mix_mag, target in train_loader:
    print("Mix magnitude shape:", mix_mag.shape)
    print("Target shape:", target.shape)
    break  # Just print first batch

'''
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    learning_rate = 0.0001
    epochs = 20
    weight_decay = 1e-4

    model = ConvUNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                predicted = torch.argmax(output.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

    # save the model
    torch.save(model.state_dict(), 'song_splitter_model.pth')

if __name__ == '__main__':
    main()'''