import openai
import requests
from PIL import Image
import io

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

def handle_image_command(command):
    if "создай изображение" in command.lower():
        image_description = command.lower().split("создай изображение")[1].strip()
        if image_description:
            img = get_image_from_dalle(image_description)
            if img:
                print("Изображение создано и показано.")
            else:
                print("Не удалось создать изображение.")
        else:
            print("Необходимо указать описание изображения после команды 'создай изображение'.")
