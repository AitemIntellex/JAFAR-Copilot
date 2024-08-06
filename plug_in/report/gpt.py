import openai
from jafar.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

prompt = "Hello, how can I use OpenAI API to connect to ChatGPT?"
response = get_chatgpt_response(prompt)
print(response)
