import speech_recognition as sr


recognizer = sr.Recognizer()

def convert_speech_to_text(filename):
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    return recognizer.recognize_google(audio)


if __name__ == "__main__":
    filename = "emotion-speech-dataset/augmented/remember.wav"
    text = convert_speech_to_text(filename)
    print(text)
