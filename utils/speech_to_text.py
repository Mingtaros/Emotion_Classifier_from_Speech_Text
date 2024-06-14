import librosa
from transformers import pipeline
from device_utils import check_gpu


device = check_gpu()
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)


def convert_speech_to_text(pipe, filename, sampling_rate=48000):
    wave, _ = librosa.load(filename, sr=sampling_rate)
    result = pipe({
        "array": wave,
        "sampling_rate": sampling_rate
    })
    return result["text"]


if __name__ == "__main__":
    filename = "emotion-speech-dataset/augmented/remember.wav"
    text = convert_speech_to_text(pipe, filename)
    print(text)
