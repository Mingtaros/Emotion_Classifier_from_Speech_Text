import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

from device_utils import check_gpu
# from speech_to_text import pipe, convert_speech_to_text
# evaluate the overall system

# PREPROCESSORS
scaler = joblib.load("./models/scaler.pkl")

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result


with open("./models/vocab2index.json", 'r') as f:
    vocab2index = json.load(f)

def encode_sentence(text, vocab2index, max_len=128):
    encoded = np.zeros(max_len, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
    length = min(max_len, len(enc1)) # if above max len, cut the rest
    encoded[:length] = enc1[:length]

    return encoded


# PREDICTION MODELS
emotions_dict = {
    0: 'surprised',
    1: 'neutral',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust'
}

SPEECH_MODEL_PATH = "models/speech_model.keras"
TEXT_MODEL_PATH = "models/torch_text_cnn_model_2024.06.15.19.56.37.pth"
device = check_gpu()

class ConvolutionalModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(ConvolutionalModel, self).__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim=128, padding_idx=1)
        self.conv1 = nn.Conv1d(128, 64, 3)
        self.conv2 = nn.Conv1d(64, 32, 3)
        self.dropout1 = nn.Dropout(0.1)
        self.linear_size = 32 * 124
        self.linear1 = nn.Linear(self.linear_size, 64)
        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(64, output_size)

    
    def forward(self, input_text):
        embedded = self.embedding(input_text)
        # embedded = [batch size, seq len, embedding dim]
        # need to convert to:
        # embedded = [batch size, embedding dim, seq len]
        embedded = embedded.permute(0, 2, 1)
        output = F.relu(self.conv1(embedded))
        output = F.relu(self.conv2(output))
        output = self.dropout1(output).view(-1, self.linear_size)
        output = F.relu(self.linear1(output))
        output = self.dropout2(output)
        output = self.linear2(output) # no need softmax

        return output

speech_model = tf.keras.models.load_model(SPEECH_MODEL_PATH)

vocab_size = 17720
text_model = ConvolutionalModel(vocab_size, output_size=len(emotions_dict)).to(device)
text_model.load_state_dict(torch.load(TEXT_MODEL_PATH))
text_model.eval()


def predict(speech_model, text_model, wav_file):
    # speech model predict
    X = []
    speech_feature = get_features(wav_file)
    for ele in speech_feature:
        X.append(ele)
    
    Features = np.array(X)
    Features = Features.reshape(-1, len(Features))
    scaled_features = scaler.transform(Features)
    predicted = speech_model.predict(scaled_features)
    emotion_by_speech = tf.argmax(predicted, axis=1)

    # text_from_speech = convert_speech_to_text(pipe, wav_file)
    text_from_speech = "Remember to Submit your Assignment"
    encoded_text = np.array(encode_sentence(text_from_speech, vocab2index, max_len=128))
    encoded_text = encoded_text.reshape(-1, len(encoded_text))
    predicted = text_model(torch.tensor(encoded_text).to(device))
    _, emotion_by_text = torch.max(predicted.data, axis=1)

    return emotion_by_speech[0].numpy(), emotion_by_text.cpu().numpy()[0]


def eval():
    pass


if __name__ == "__main__":
    wav_file = "emotion-speech-dataset/augmented/remember.wav"

    emotion_by_speech, emotion_by_text = predict(speech_model, text_model, wav_file)

    print(emotion_by_speech, emotion_by_text)
    print("Speech:", emotions_dict[emotion_by_speech])
    print("Text  :", emotions_dict[emotion_by_text])
