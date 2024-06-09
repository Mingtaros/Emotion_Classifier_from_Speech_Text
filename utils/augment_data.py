import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from tqdm.notebook import tqdm

# Define your augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

def load_audio(file_path, sample_rate=48000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y, sr

def augment_audio(y, sample_rate):
    y_augmented = augment(samples=y, sample_rate=sample_rate)
    return y_augmented

def concatenate_audio(y_original, y_augmented):
    return np.concatenate((y_original, y_augmented))

def augment_and_concatenate_audio(input_dir, output_dir, sample_rate=48000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dir_list = os.listdir(input_dir)
    dir_list.sort()
    
    for i in tqdm(dir_list, total=len(dir_list)):
        if i == ".DS_Store":
            continue

        if not i.startswith("Actor_"):
            continue
        
        fname = os.listdir(input_dir+i)
        
        if not os.path.exists(output_dir+i):
            os.makedirs(output_dir+i)
        
        for f in fname:
            part = f.split('.')[0].split('-')
            file_path = input_dir + i + '/' + f
            
            y_original, sr = load_audio(file_path, sample_rate=sample_rate)
            y_augmented = augment_audio(y_original, sr)
            y_concatenated = concatenate_audio(y_original, y_augmented)

            output_file_path = output_dir + i + '/' + f
            sf.write(output_file_path, y_concatenated, sr)
            print(f"Saved augmented file to {output_file_path}")

# Example usage
input_dir = './data/ravdess-emotional-speech-audio/'
output_dir = './data/ravdess-emotional-speech-audio-augmented/'
augment_and_concatenate_audio(input_dir, output_dir)