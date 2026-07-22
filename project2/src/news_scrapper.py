import requests

NEWS_API_KEY = "e3c0922b09ad467fbdc24dc02babe42a"

def fetch_live_news(query=None, category=None, country="us", max_results=10):
    """
    Fetches articles from NewsAPI.
    """
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "pageSize": max_results
    }
    
    if query:
        url = "https://newsapi.org/v2/everything"
        params["q"] = query
    else:
        if country:
            params["country"] = country
        if category:
            params["category"] = category
            
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            
            cleaned_articles = []
            for art in articles:
                title = art.get("title", "")
                desc = art.get("description", "")
                content = art.get("content", "")
                full_text = f"{title}. {desc or ''} {content or ''}".strip()
                
                cleaned_articles.append({
                    "title": title,
                    "description": desc,
                    "source": art.get("source", {}).get("name", "Unknown"),
                    "url": art.get("url", ""),
                    "full_text": full_text
                })
            return cleaned_articles
        else:
            print(f"Error fetching news: API returned status {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception during news fetch: {e}")
        return []
