from datamanagement.scraping.community import CommunityScraper
from datamanagement.scraping.webex import WebexScraper
from datamanagement.scraping.base import BaseScraper
from datamanagement.core.logger import setup_logger

logger = setup_logger('scraper_factory', 'datamanagement/log/scraper_factory.log')

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
        logger.info(f"Request to get scraper for source='{source}', url='{url}'")
        scraper_cls = ScraperFactory._registry.get(source.lower())
        if not scraper_cls:
            logger.error(f"Unsupported source type requested: {source}")
            raise ValueError(f"Unsupported source type: {source}")
        logger.info(f"Instantiating scraper class: {scraper_cls.__name__}")
        return scraper_cls(source, url)
