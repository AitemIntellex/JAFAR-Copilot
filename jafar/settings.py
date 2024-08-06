# settings.py
import openai
import speech_recognition as sr
from PIL import Image
import requests
import pygame
import tempfile
import io
from dotenv import load_dotenv
import os

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация pygame для воспроизведения звуков
pygame.mixer.init()

# Установка ключа API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
