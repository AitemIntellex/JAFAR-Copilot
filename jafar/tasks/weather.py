import requests
from config import OWM_API_KEY
from tasks.general import text_to_speech

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_API_KEY}&units=metric&lang=ru"
    print(f"Запрос к OpenWeatherMap: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Ответ от OpenWeatherMap: {data}")
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"Погода в {city}: {weather}, температура: {temp}°C"
    else:
        print(f"Ошибка при запросе к OpenWeatherMap: {response.status_code} {response.text}")
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
