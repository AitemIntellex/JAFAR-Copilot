import requests
from config import NEWS_API_KEY

def get_latest_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        return articles
    else:
        return None

def handle_news_command():
    articles = get_latest_news()
    if articles:
        news_summary = "\n".join([f"{article['title']}" for article in articles[:5]])
        return news_summary
    else:
        return "Не удалось получить новости."
