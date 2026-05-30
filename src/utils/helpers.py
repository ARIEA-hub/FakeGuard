import logging
import time
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_MAX_CONTENT_CHARS = 10_000


def scrape_article_from_url(
    url: str, timeout: int = 10
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch article title and body text from a URL.
    Returns (title, text) — either may be None on failure.
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching %s", url)
        return None, None
    except requests.exceptions.HTTPError as exc:
        logger.error("HTTP %s for %s", exc.response.status_code, url)
        return None, None
    except requests.exceptions.RequestException as exc:
        logger.error("Request error for %s: %s", url, exc)
        return None, None

    try:
        soup = BeautifulSoup(response.text, "lxml")

        # Extract title
        title = ""
        if h1 := soup.find("h1"):
            title = h1.get_text(strip=True)
        elif og_title := soup.find("meta", property="og:title"):
            title = og_title.get("content", "").strip()
        elif t := soup.find("title"):
            title = t.get_text(strip=True)

        # Remove boilerplate nodes
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe", "noscript", "figure"]):
            tag.decompose()

        # Prefer <article> semantic block; fall back to <p> accumulation
        article_tag = soup.find("article")
        if article_tag:
            body = article_tag.get_text(separator=" ", strip=True)
        else:
            paragraphs = [
                p.get_text(strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 40
            ]
            body = " ".join(paragraphs)

        if not body or len(body.split()) < 20:
            logger.warning("Insufficient text extracted from %s", url)
            return title or None, None

        return title or None, body[:_MAX_CONTENT_CHARS]

    except Exception as exc:
        logger.error("Parse error for %s: %s", url, exc)
        return None, None


def rate_limiter(last_call_ts: float, min_interval: float = 1.0) -> bool:
    """Return True if enough time has elapsed since the last call."""
    return (time.monotonic() - last_call_ts) >= min_interval