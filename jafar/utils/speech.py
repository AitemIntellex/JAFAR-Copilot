# speech.py
import wave
import sys

import pyaudio
import speech_recognition as sr
from .sound import play_sound

def recognize_speech():
    recognizer = sr.Recognizer()
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Слушаю...")
        play_sound('assets/start_listening.mp3')
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"Dave Minchen: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
        return ""
    except sr.RequestError:
        print("Ошибка запроса к сервису Google Speech Recognition")
        return ""
