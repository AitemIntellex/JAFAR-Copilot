# text_to_speech.py
import openai
import requests
import tempfile
import pygame
from .sound import play_sound

def text_to_speech(text):
    play_sound('assets/start_speaking.mp3')
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
