import speech_recognition as sr
import pygame
import json
import vosk
import pyaudio

recognizer = sr.Recognizer()

def recognize_speech():
    with sr.Microphone() as source:
        print("Слушаю...")
        pygame.mixer.music.load('jafar/start_listening.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"Вы сказали: {text}")
        return text
    except sr.UnknownValueError:
        print("Не удалось распознать речь. Попробуйте снова.")
        return recognize_speech()  # Повторный запрос
    except sr.RequestError:
        print("Ошибка запроса к сервису Google Speech Recognition")
        return ""

def detect_keyword(model):
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                print(f"Распознанный текст: {result['text']}")
                if "джафар" in result['text'].lower():
                    return True
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return False
