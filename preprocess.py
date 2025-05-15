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

def preprocess_audio(filename, time, fft_size=2048, hop_length=512, save_dir='processed_stems'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # read audio
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
    
    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(save_dir, base_name + '.pt')
    torch.save(processed_stems, save_path)
    print(f"Saved processed stems to {save_path}")

if __name__ == '__main__':
    train_dir = '/root/song-splitter/musdb18/train'
    for file in os.listdir(train_dir):
        if file.endswith('.stem.mp4'):  # Only process stem files
            full_path = os.path.join(train_dir, file)
            print(f"Processing {file}...")
            preprocess_audio(full_path, 30)