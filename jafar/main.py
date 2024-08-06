import signal
import sys
import os

# Добавляем путь к модулям plug_in
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plug_in', 'report')))

from voice_recognition import detect_keyword, recognize_speech
from response_generation import get_response
from voice_output import text_to_speech
from tasks.weather import handle_weather_command
from tasks.general import handle_general_command, text_to_speech
import vosk

model = vosk.Model(r"jafar/model/vosk-model-small-ru-0.22")

def signal_handler(sig, frame):
    print("Завершение работы...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("Готов к работе! Скажите 'Джафар' для активации ассистента.")

    while True:
        if detect_keyword(model):
            print("Ассистент активирован. Слушаю команду...")
            command = recognize_speech(model)
            if command:
                response = get_response(command)
                print(f"Джафар: {response}")
                text_to_speech(response)
                if "отбой" in command.lower():
                    print("Ассистент приостановлен. Скажите 'Джафар' для повторной активации.")

if __name__ == "__main__":
    main()
