import json
import glob
import nltk
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

from device_utils import check_gpu
from speech_to_text import convert_speech_to_text


# initialize necessary models
JUSTIN_REFERENCE_FILE = "./justin_recording/justin_2024.csv"
SCALER_PATH = "./models/scaler.pkl"
SPEECH_MODEL_PATH = "./models/best_speech_model.pth"
SPEECH_MODEL_POSNEG_PATH = "./models/best_posneg_speech_model.pth"
# TEXT_MODEL_PATH = "./models/torch_text_linear_model_2024.06.20.15.39.49.pth"
TEXT_MODEL_PATH = "./models/torch_text_cnn_model_2024.06.20.12.20.41.pth"
TEXT_MODEL_TYPE = "pytorch" # change to `pytorch` if using pytorch model, `keras` for keras model
VOCAB2INDEX_PATH = "./models/vocab2index_built.json"

MAX_ENCODED_LEN = 20
# MAX_ENCODED_LEN = 70

# read data
justin_reference = pd.read_csv(JUSTIN_REFERENCE_FILE)
# remove disgust
# justin_reference = justin_reference[justin_reference["Emotion"] != "disgust"]


# PREPROCESSORS
scaler = joblib.load(SCALER_PATH)
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
    features = []

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    features.append(zcr)

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    features.append(chroma_stft)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    features.append(mfcc)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    features.append(rms)

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    features.append(mel)
    
    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    features.append(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
    features.append(spectral_bandwidth)

    # Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    features.append(spectral_contrast)

    # Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
    features.append(spectral_rolloff)

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sample_rate).T, axis=0)
    features.append(tonnetz)
    
    return np.hstack(features)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result


# text preprocessing
def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = word_tokenize(text)
    # remove stopwords
    text = [token for token in text if token not in stop_words]
    # lemmatizer
    text = [lemmatizer.lemmatize(token) for token in text]

    # return detokenizer.detokenize(text).strip()
    return text

with open(VOCAB2INDEX_PATH, 'r') as f:
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

reverse_emotions_dict = {
    'surprised': 0,
    'neutral': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6
}

emotion_to_posneg = {
    'surprised': 1,
    'neutral': 1,
    'happy': 1,
    'sad': 0,
    'angry': 0,
    'fearful': 0
}

num_to_posneg = {
    0: "negative",
    1: "positive"
}

posneg_to_num = {
    "negative": 0,
    "positive": 1
}

device = check_gpu()

class CNN_LSTMModel(nn.Module):
    def __init__(self):
        super(CNN_LSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Adjust LSTM input size based on the output from CNN layers
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        
        # Calculate the input size for the fully connected layer
        self.fc1 = nn.Linear(2944, 128)  # Adjust the input dimension based on your data
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # Adjust output dimension based on the number of classes

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, input_size)
        x, _ = self.lstm(x)
        
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout4(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)  # No Softmax here
        return x

speech_model = CNN_LSTMModel().to(device)
speech_model.load_state_dict(torch.load(SPEECH_MODEL_PATH, map_location=device))
speech_model.eval()

class SimpleLinearModel(nn.Module):
    def __init__(self, vocab_size, input_size, output_size, embedding_matrix=None, freeze_embeddings=True):
        super(SimpleLinearModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = 100

        self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.linear_size = input_size * self.embedding_dim
        self.linear1 = nn.Linear(self.linear_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(64, output_size)


    def forward(self, inputs):
        # we assume the inputs already in embedding dimension
        output = self.embedding(inputs).view(-1, self.linear_size)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.dropout1(output)
        output = self.linear4(output)

        return output

class ConvolutionalModel(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(ConvolutionalModel, self).__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim=128, padding_idx=1)
        self.conv1 = nn.Conv1d(128, 64, 3)
        self.conv2 = nn.Conv1d(64, 32, 3)
        self.dropout1 = nn.Dropout(0.1)
        self.linear_size = 32 * 16
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
        output = self.linear2(output) # no need sigmoid

        return output


vocab_size = len(vocab2index) # + 2 for "UNK" and ""
if TEXT_MODEL_TYPE == "keras":
    text_model = tf.keras.models.load_model(TEXT_MODEL_PATH)
else:
    # text_model = SimpleLinearModel(vocab_size, input_size=MAX_ENCODED_LEN, output_size=1).to(device)
    text_model = ConvolutionalModel(vocab_size, output_size=1).to(device)
    text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
    text_model.eval()


def predict(speech_model, text_model, wav_file, text_in_wav=None):
    # speech model predict
    X = []
    speech_feature = get_features(wav_file)
    X.extend(speech_feature)
    
    Features = np.array(X)
    Features = Features.reshape(-1, len(Features))
    scaled_features = scaler.transform(Features)
    scaled_features = np.expand_dims(Features, axis=1)

    # speech
    predicted = speech_model(torch.tensor(scaled_features, dtype=torch.float32).to(device))
    emotion_proba, emotion_by_speech = torch.max(F.softmax(predicted.data, dim=1), axis=1)
    emotion_by_speech = emotion_by_speech.cpu().numpy()[0]
    # map it to either positive or negative
    emotion_by_speech = emotion_to_posneg[emotions_dict[emotion_by_speech]]
    if emotion_by_speech == 1:
        speech_proba = emotion_proba
    else:
        speech_proba = 1 - emotion_proba
    
    speech_proba = speech_proba.cpu().numpy()[0]

    # text
    if not text_in_wav:
        # if there's no text, extract
        text_in_wav = convert_speech_to_text(wav_file)
    
    text_in_wav = clean_text(text_in_wav)
    text_in_wav = encode_sentence(text_in_wav, vocab2index, max_len=MAX_ENCODED_LEN)

    encoded_text = np.array(text_in_wav)
    encoded_text = encoded_text.reshape(-1, len(encoded_text))

    predicted = text_model(torch.tensor(encoded_text).to(device))
    predicted = predicted.squeeze(-1)
    emotion_by_text = torch.round(F.sigmoid(predicted.data))
    emotion_by_text = int(emotion_by_text.cpu().numpy()[0])
    text_proba = F.sigmoid(predicted.data).cpu().numpy()[0]

    return emotion_by_speech, speech_proba, emotion_by_text, text_proba


def confusion_matrix_visualize(conf_mat, title, filename, index_column_values=emotions_dict.values()):
    cm = pd.DataFrame(conf_mat, index=index_column_values, columns=index_column_values)
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    # plt.show()
    plt.savefig(filename)
    plt.close()


def combined_model_pred(speech_proba, text_proba, cutoff, speech_weight=1, text_weight=1):
    final_proba = (speech_proba * speech_weight) + (text_proba * text_weight)
    final_proba /= (speech_weight + text_weight)

    if final_proba >= cutoff:
        return 1
    else:
        return 0


def eval(speech_model, text_model):
    # evaluate the performance of the models on justin recordings

    prediction_result = []

    for justin_filename, justin_text, justin_emotion_pos_neg in justin_reference[["FileName", "Text", "Emotion_Pos_Neg"]].values:
        # get readable filename
        justin_speech = f"justin_recording/{justin_filename}.wav"
        posneg_label = justin_emotion_pos_neg
        posneg_label_encoded = posneg_to_num[posneg_label]

        print(justin_speech)
        speech_pred, speech_proba, text_pred, text_proba = predict(speech_model, text_model, justin_speech, justin_text)

        prediction_result.append({
            "filename": justin_speech.split("/")[-1],
            "posneg_label": posneg_label,
            "posneg_label_encoded": posneg_label_encoded,
            "speech_prediction": num_to_posneg[speech_pred],
            "speech_prediction_encoded": speech_pred,
            "speech_proba": speech_proba,
            "text_prediction": num_to_posneg[text_pred],
            "text_prediction_encoded": text_pred,
            "text_proba": text_proba,
        })
    
    prediction_result = pd.DataFrame(prediction_result)
    # get performance measures
    speech_report = classification_report(prediction_result["posneg_label_encoded"], prediction_result["speech_prediction_encoded"], target_names=posneg_to_num.keys())
    speech_conf_mat = confusion_matrix(prediction_result["posneg_label_encoded"], prediction_result["speech_prediction_encoded"])

    text_report = classification_report(prediction_result["posneg_label_encoded"], prediction_result["text_prediction_encoded"], target_names=posneg_to_num.keys())
    text_conf_mat = confusion_matrix(prediction_result["posneg_label_encoded"], prediction_result["text_prediction_encoded"])

    combined_pred = [
        combined_model_pred(speech_proba, text_proba, cutoff=0.5)
        for speech_proba, text_proba
        in prediction_result[["speech_proba", "text_proba"]].values
    ]

    combined_report = classification_report(prediction_result["posneg_label_encoded"], combined_pred, target_names=posneg_to_num.keys())
    combined_conf_mat = confusion_matrix(prediction_result["posneg_label_encoded"], combined_pred)

    print("=======Speech Model Performance=======")
    print(speech_report)
    print("========Text Model Performance========")
    print(text_report)
    print("======Combined Model Performance======")
    print(combined_report)

    confusion_matrix_visualize(speech_conf_mat, "Speech Model Confusion Matrix", "speech_conf_mat.png", posneg_to_num.keys())
    confusion_matrix_visualize(text_conf_mat, "Text Model Confusion Matrix", "text_conf_mat.png", posneg_to_num.keys())
    confusion_matrix_visualize(combined_conf_mat, "Combined Model Confusion Matrix", "combined_conf_mat.png", posneg_to_num.keys())
    
    # save prediction result
    prediction_result.to_csv("justin_prediction_result.csv", index=False)


if __name__ == "__main__":
    eval(speech_model, text_model)
