import speech_recognition as sr
import pygame
import json
import vosk
import pyaudio
import numpy as np
from scipy.signal import butter, lfilter

recognizer = sr.Recognizer()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def recognize_speech():
    with sr.Microphone() as source:
        print("Слушаю...")
        pygame.mixer.music.load('start_listening.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"Вы сказали: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
        return ""
    except sr.RequestError:
        print("Ошибка запроса к сервису Google Speech Recognition")
        return ""

def detect_keyword(model):
    print("Запуск Vosk для обнаружения ключевого слова...")
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            filtered_data = bandpass_filter(audio_data, 300.0, 3400.0, 16000, order=6)
            filtered_data = filtered_data.astype(np.int16).tobytes()
            if rec.AcceptWaveform(filtered_data):
                result = json.loads(rec.Result())
                print(f"Распознанный текст: {result['text']}")
                if "джафар" in result['text'].lower():
                    print("Ключевое слово обнаружено. Активирую ассистента...")
                    return True
    except KeyboardInterrupt:
        print("Остановка потока обнаружения ключевого слова")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return False
