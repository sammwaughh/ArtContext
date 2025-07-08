import requests
import pandas as pd

def fetch_topics():
    topics = []
    per_page = 200
    page = 1
    while True:
        url = f"https://api.openalex.org/topics?per_page={per_page}&page={page}"
        print(f"Fetching page {page}")
        response = requests.get(url)
        data = response.json()
        results = data.get("results", [])
        if not results:
            break
        topics.extend(results)
        # If the number of results is less than per_page, we've reached the end.
        if len(results) < per_page:
            break
        page += 1
    return topics

if __name__ == "__main__":
    topics = fetch_topics()
    print(f"Fetched {len(topics)} topics")
    df = pd.DataFrame(topics)
    df.to_excel("openalex_topics.xlsx", index=False, engine="openpyxl")
    print("Saved topics to openalex_topics.xlsx")
