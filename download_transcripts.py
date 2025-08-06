import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://datatalks.club/podcast.html"
EPISODES_DIR = "data/raw"

def fetch_episode_links():
    print("🔍 Fetching episode links...")
    resp = requests.get(BASE_URL)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/podcast/s"):
            full_url = "https://datatalks.club" + href
            links.append(full_url)

    print(f"✅ Found {len(links)} links.")
    return sorted(set(links))

def download_transcript(url):
    print(f"🔽 Downloading: {url}")
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text().strip().replace(" ", "_") if title_tag else "unknown_episode"

        transcript_header = soup.find("h2", string="Transcript")
        if not transcript_header:
            print(f"⚠️ Episode {title} does not contain a transcript section.")
            return

        content = []
        for elem in transcript_header.find_all_next():
            if elem.name and elem.name.startswith("h2") and elem.get_text().strip().lower() in [
                "resources", "subscribe", "links"
            ]:
                break
            if elem.name in ["p", "ul", "ol", "div"]:
                text = elem.get_text().strip()
                if text:
                    content.append(text)

        if not content:
            print(f"⚠️ Episode {title} contains no useful transcript content.")
            return

        text = "\n\n".join(content)
        os.makedirs(EPISODES_DIR, exist_ok=True)
        filename = os.path.join(EPISODES_DIR, f"{title}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"💾 Saved: {filename}")
    except Exception as e:
        print(f"❌ Error processing {url}: {e}")

def main():
    links = fetch_episode_links()
    for link in links:
        download_transcript(link)

if __name__ == "__main__":
    main()
