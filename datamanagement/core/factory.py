from datamanagement.scraping.community import CommunityScraper
from datamanagement.scraping.webex import WebexScraper
from datamanagement.scraping.base import BaseScraper

class ScraperFactory:
    """
    Factory to return the appropriate scraper subclass instance
    based on the source identifier.
    """
    _registry = {
        "community": CommunityScraper,
        "webex": WebexScraper,
    }

    @staticmethod
    def get_scraper(source: str, url: str) -> BaseScraper:
        scraper_cls = ScraperFactory._registry.get(source.lower())
        if not scraper_cls:
            raise ValueError(f"Unsupported source type: {source}")
        return scraper_cls(source, url)
