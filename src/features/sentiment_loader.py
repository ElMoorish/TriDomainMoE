"""
Yahoo Finance News and Macro Article Harvester.
Pulls real-time headlines and summaries with deterministic publication timestamp tracking.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import email.utils
import logging
import requests

logger = logging.getLogger(__name__)


class YahooFinanceSentimentLoader:
    """Fetches market news and macro articles via Yahoo Finance feeds and APIs."""

    DEFAULT_RSS = "https://finance.yahoo.com/news/rssindex"
    SYMBOL_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def fetch_articles(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch article metadata from RSS feeds.
        Returns list of dicts with: ['title', 'link', 'published_at', 'summary']
        """
        url = self.SYMBOL_RSS_TEMPLATE.format(ticker=ticker) if ticker else self.DEFAULT_RSS
        logger.info("Fetching Yahoo Finance news from %s", url)

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning("Yahoo Finance RSS returned status code %d", resp.status_code)
                return []

            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            articles = []

            for item in items:
                title_elem = item.find("title")
                link_elem = item.find("link")
                pub_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                summary = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

                published_at = datetime.now(timezone.utc)
                if pub_elem is not None and pub_elem.text:
                    try:
                        parsed_tuple = email.utils.parsedate_tz(pub_elem.text)
                        if parsed_tuple:
                            published_at = datetime.fromtimestamp(email.utils.mktime_tz(parsed_tuple), tz=timezone.utc)
                    except Exception:
                        pass

                articles.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published_at": published_at,
                    "ticker": ticker,
                })

            logger.info("Retrieved %d articles for ticker %s", len(articles), ticker or "GLOBAL")
            return articles

        except Exception as exc:
            logger.error("Error fetching Yahoo Finance news: %s", exc)
            return []
