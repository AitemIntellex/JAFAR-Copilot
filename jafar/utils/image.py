# image.py
import openai
import requests
from PIL import Image
import io

def get_image_from_dalle(prompt):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "model": "image-alpha-001",
        "num_images": 1
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        image_url = response.json()["data"][0]["url"]
        image_response = requests.get(image_url)
        img = Image.open(io.BytesIO(image_response.content))
        img.show()
        return img
    else:
        print(f"Ошибка создания изображения: {response.status_code} {response.json()}")
        return None
