import openai
import sys
import os
from datetime import datetime
import pytz

# Добавляем путь к модулям plug_in
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'plug_in')))



def handle_general_command(command: str):
    if "создать событие" in command.lower():
        return create_event(command)
    elif "показать события" in command.lower():
        return show_events()
    else:
        return "Команда не распознана."

def create_event(command: str):
    # Пример простого создания события
    # Реализуйте более сложную логику в зависимости от ваших требований
    event_title = "Встреча"
    event_time = datetime.now(pytz.timezone('Europe/Moscow'))
    event = {
        "title": event_title,
        "time": event_time.isoformat()
    }
    # Сохраните событие в базе данных или другом хранилище
    return f"Событие '{event_title}' создано на {event_time.strftime('%Y-%m-%d %H:%M:%S')}."

def show_events():
    # Получите список событий из базы данных или другого хранилища
    events = [
        {"title": "Встреча", "time": "2024-06-20T15:00:00+03:00"},
        {"title": "Совещание", "time": "2024-06-21T10:00:00+03:00"}
    ]
    events_list = "\n".join([f"{event['title']} - {event['time']}" for event in events])
    return f"Список событий:\n{events_list}"

def text_to_speech(text):
    # Логика преобразования текста в речь
    import pygame
    import tempfile
    import requests

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
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name

        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    else:
        print(f"Ошибка преобразования текста в речь: {response.status_code} {response.json()}")
