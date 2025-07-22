"""
Module: base_scraper.py

Defines an abstract base class for all scraper implementations.
Handles common scraper initialization and provides a utility method
for HTML content retrieval via URLAccess.

Subclasses must implement the scrape() method to extract relevant data.
"""
from abc import ABC, abstractmethod
from datamanagement.scraping.url_access import URLAccess
from datamanagement.core.logger import setup_logger

logger = setup_logger('base_scraper', 'datamanagement/log/base_scraper.log')

class BaseScraper(ABC):
    """
    Abstract base class for scrapers.

    Attributes:
        source (str): Identifier for the data source (e.g., 'community', 'webex').
        url (str): URL of the page to be scraped.
    """
    def __init__(self, source, url):
        """
        Initialize the base scraper with source and URL.

        Args:
            source (str): Source identifier.
            url (str): Target URL for scraping.
        """
        self.source = source
        self.url = url
        logger.info(f"Initialized BaseScraper with source={source} and url={url}")

    def _get_soup(self):
        """
        Retrieves and returns the BeautifulSoup content of the target URL.

        Returns:
            BeautifulSoup: Parsed HTML content.

        Raises:
            Exception: If content retrieval fails.
        """
        try:
            logger.debug(f"Fetching content for URL {self.url}")
            content = URLAccess(self.source, self.url).content()
            logger.info(f"Content successfully fetched for URL {self.url}")
            return content
        except Exception as e:
            logger.error(f"Failed to get content for URL {self.url}: {e}")
            raise


    @abstractmethod
    def scrape(self):
        """
        Abstract method to be implemented by subclasses to parse and extract data from the page.

        Raises:
            NotImplementedError: If the method is not implemented by subclass.
        """
        pass
