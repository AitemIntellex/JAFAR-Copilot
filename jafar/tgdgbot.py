import openai
import speech_recognition as sr
import pygame
import requests
import tempfile
import os
import vosk
import pyaudio
import json
import asyncio
from tgdbot import initialize_mt5, smc_trading_strategy, get_account_info

import os
# Отладочный код


# Инициализация pygame для воспроизведения звуков
pygame.mixer.init()

# Переменная для состояния ассистента
is_active = False

# Инициализация модели Vosk
model = vosk.Model(r"d:\\model\vosk-model-small-ru-0.22")

def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Слушаю...")
        play_sound('start_listening.mp3')
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

def text_to_speech(text):
    play_sound('start_speaking.mp3')
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "tts-1",
        "voice": "fable"
    }
    print(f"Отправка запроса на преобразование текста в речь: {text}")
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name
        print(f"Текст преобразован в речь и сохранен как {temp_audio_file_path}")

        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    else:
        print(f"Ошибка преобразования текста в речь: {response.status_code} {response.json()}")

async def handle_trading_command(command):
    if "открой позицию" in command.lower():
        # Запуск стратегии для всех символов
        await smc_trading_strategy("XAUUSDm", 0.01)  # Пример для символа XAUUSDm
        await text_to_speech("Позиция открыта.")

    elif "статус счета" in command.lower():
        account_info = get_account_info()
        if account_info:
            await text_to_speech(f"Информация о счете: {account_info}")
        else:
            await text_to_speech("Не удалось получить информацию о счете.")

    elif "закрой позицию" in command.lower():
        # Логика закрытия позиций через торгового бота
        await text_to_speech("Закрытие позиции не реализовано. Мы добавим эту функцию позже.")

    else:
        await text_to_speech("Команда для торговли не распознана.")

def handle_command(command):
    global is_active
    print(f"Обработка команды: {command}")

    if "отбой" in command.lower():
        is_active = False
        print("Ассистент приостановлен.")
        text_to_speech("До свидания.")

    elif "трейдинг" in command.lower():
        asyncio.run(handle_trading_command(command))

    else:
        response = get_openai_response(command)
        print(f"Джафар: {response}")
        text_to_speech(response)

def detect_keyword():
    global is_active
    print("Запуск Vosk для обнаружения ключевого слова...")
    rec = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if "джафар" in result['text'].lower():
                    print("Ключевое слово обнаружено. Активирую ассистента...")
                    is_active = True
                    break
    except KeyboardInterrupt:
        print("Остановка потока обнаружения ключевого слова")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    global is_active
    print("Готов к работе! Скажите 'Джафар' для активации ассистента.")

    while True:
        if not is_active:
            detect_keyword()
        else:
            print("Ассистент активен. Слушаю команду...")
            command = recognize_speech()
            if command:
                handle_command(command)

if __name__ == "__main__":
    main()
