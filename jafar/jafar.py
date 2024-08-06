import openai
import speech_recognition as sr
from config import OWM_API_KEY
from PIL import Image
import requests
import pygame
import tempfile
import io
import json
import wave
import vosk
import pyaudio
import signal
import sys
from dotenv import load_dotenv
import os
# Отладочный код
print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("Virtual environment:", os.environ.get('VIRTUAL_ENV'))

# Инициализация pygame для воспроизведения звуков
pygame.mixer.init()

# Переменная для состояния ассистента
is_active = False

# Инициализация модели Vosk
model = vosk.Model(r"jafar\model\vosk-model-small-ru-0.22")

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
        "voice": "onyx"
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

def get_openai_response(prompt):
    print(f"Отправка запроса в OpenAI: {prompt}")
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=220
    )
    answer = response.choices[0].message["content"].strip()
    print(f"Ответ от OpenAI: {answer}")
    return answer

def get_image_from_dalle(prompt):
    if not prompt:
        print("Описание изображения пустое.")
        return None

    print(f"Отправка запроса на создание изображения в DALL-E: {prompt}")
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    image_response = requests.get(image_url)
    img = Image.open(io.BytesIO(image_response.content))
    img.show()
    return img

def process_image(image_path):
    with Image.open(image_path) as img:
        img.show()
        img_resized = img.resize((100, 100))
        img_resized.show()
        print("Изображение обработано")

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_API_KEY}&units=metric&lang=ru"
    print(f"Запрос к OpenWeatherMap: {url}")  # Добавлено для отладки
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Ответ от OpenWeatherMap: {data}")  # Добавлено для отладки
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Погода в {city}: {weather}, температура: {temp}°C"
    else:
        print(f"Ошибка при запросе к OpenWeatherMap: {response.status_code} {response.text}")  # Добавлено для отладки
        return "Не удалось получить данные о погоде."

def handle_weather_command(command):
    try:
        if "какая погода в городе" in command.lower():
            city = command.lower().split("какая погода в городе")[1].strip()
            if city:
                weather_info = get_weather(city)
                print(f"Джафар: {weather_info}")
                text_to_speech(weather_info)
            else:
                print("Необходимо указать город после команды 'какая погода в'.")
        else:
            print("Команда для получения погоды не распознана.")
    except IndexError:
        print("Ошибка в разборе команды для получения погоды. Убедитесь, что команда правильно сформулирована.")

def handle_command(command):
    global is_active
    print(f"Обработка команды: {command}")

    if "Создай изображение" in command.lower():
        image_description = command.lower().split("Создай изображение")[1].strip()
        if image_description:
            img = get_image_from_dalle(image_description)
            if img:
                print("Изображение создано и показано.")
            else:
                print("Не удалось создать изображение.")
        else:
            print("Необходимо указать описание изображения после команды 'создай изображение'.")

    elif "отбой" in command.lower():
        is_active = False
        print("Ассистент приостановлен.")
        text_to_speech("До свидания.")

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

def signal_handler(sig, frame):
    print("Завершение работы...")
    sys.exit(0)

def main():
    global is_active
    signal.signal(signal.SIGINT, signal_handler)
    print("Готов к работе! Скажите 'Джафар' для активации ассистента.")

    while True:
        if not is_active:
            detect_keyword()
        else:
            print("Ассистент активен. Слушаю команду...")
            command = recognize_speech()
            if command:
                if "какая погода в" in command.lower():
                    handle_weather_command(command)  # Обработка команды погоды
                elif "отбой" in command.lower():
                    handle_command(command)  # Обработка команды отбой
                else:
                    response = get_openai_response(command)  # Отправка остальных команд в OpenAI
                    print(f"Джафар: {response}")
                    text_to_speech(response)
            if not is_active:
                print("Ассистент приостановлен. Скажите 'Джафар' для повторной активации.")

if __name__ == "__main__":
    main()
