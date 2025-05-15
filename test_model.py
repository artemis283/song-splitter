import torch
import numpy as np
import librosa
import stempeg
import soundfile as sf
from model import ConvUNet
import os

def load_and_preprocess_audio(filename, target_time=512, target_freq=1024, fft_size=2048, hop_length=512):
    # Read the audio file
    audio, rate = stempeg.read_stems(filename, stem_id=None, sample_rate=44100)
    
    # Get the mix (first stem)
    mix = audio[0].mean(axis=1)  # Convert to mono
    
    # Compute STFT
    stft = librosa.stft(mix, n_fft=fft_size, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Convert to tensor and add batch and channel dimensions
    magnitude = torch.tensor(magnitude, dtype=torch.float32)
    
    # Handle time dimension
    current_time = magnitude.shape[1]
    if current_time > target_time:
        start_time = (current_time - target_time) // 2
        magnitude = magnitude[:, start_time:start_time + target_time]
        stft = stft[:, start_time:start_time + target_time]
    elif current_time < target_time:
        pad_time = target_time - current_time
        magnitude = torch.nn.functional.pad(magnitude, (0, pad_time))
        stft = np.pad(stft, ((0, 0), (0, pad_time)), mode='constant')
    
    # Handle frequency dimension
    current_freq = magnitude.shape[0]
    if current_freq > target_freq:
        start_freq = (current_freq - target_freq) // 2
        magnitude = magnitude[start_freq:start_freq + target_freq, :]
        stft = stft[start_freq:start_freq + target_freq, :]
    elif current_freq < target_freq:
        pad_freq = target_freq - current_freq
        magnitude = torch.nn.functional.pad(magnitude, (0, 0, 0, pad_freq))
        stft = np.pad(stft, ((0, pad_freq), (0, 0)), mode='constant')
    
    magnitude = magnitude.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return magnitude, stft, rate

def reconstruct_audio(magnitude, phase, rate, hop_length=512):
    # Convert magnitude back to complex STFT
    stft = magnitude * np.exp(1j * np.angle(phase))
    
    # Inverse STFT
    audio = librosa.istft(stft, hop_length=hop_length)
    
    return audio

def main():
    # Load the model
    model = ConvUNet(out_channels=4)
    model.load_state_dict(torch.load('song_splitter_model.pth'))
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load and preprocess test audio
    test_file = "/root/song-splitter/musdb18/test/Al James - Schoolboy Facination.stem.mp4"  
    magnitude, original_stft, rate = load_and_preprocess_audio(test_file)
    magnitude = magnitude.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(magnitude)
    
    # Move output to CPU and convert to numpy
    output = output.cpu().numpy()
    
    # Create output directory
    os.makedirs('separated_stems', exist_ok=True)
    
    # Process each stem
    stem_names = ['drums', 'bass', 'other', 'vocals']
    for i, stem_name in enumerate(stem_names):
        # Get the magnitude for this stem
        stem_magnitude = output[0, i]  # Remove batch dimension
        
        # Reconstruct audio
        stem_audio = reconstruct_audio(stem_magnitude, original_stft, rate)
        
        # Save the audio
        output_path = os.path.join('separated_stems', f'{stem_name}.wav')
        sf.write(output_path, stem_audio, rate)
        print(f"Saved {stem_name} to {output_path}")

if __name__ == '__main__':
    main() 