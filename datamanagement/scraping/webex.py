"""
Module: webex_scraper.py

Implements a scraper for Webex Help Center pages.
Extracts the page title (used as a query) and concatenates all paragraph and list item texts as response.
Depends on BaseScraper for shared web content access utilities.
"""
from scraping.base import BaseScraper
from core.logger import setup_logger

# Initialize logger for this module
logger = setup_logger('webex_scraper', 'log/webex_scraper.log')


class WebexScraper(BaseScraper):
    """
    Scraper for Cisco Webex Help Center pages.

    Extracts:
        - Query: Page title with Webex Help Center suffix removed.
        - Response: Concatenated visible text from paragraphs and list items.
    """

    def scrape(self):
        """
        Scrapes the Webex page content to retrieve a concise query and detailed response text.

        Returns:
            tuple(str or None, str or None): A tuple of (query_text, response_text)
        """
        try:
            soup = self._get_soup()

            # Extract and clean the page title
            title = soup.select_one("title")
            query = None
            if title:
                query = title.get_text(strip=True).replace(" - Webex Help Center", "")
            logger.info(f"Extracted query from title: '{query if query else 'None'}'")

            # Extract visible paragraph and list item texts
            content_elements = soup.select('p, li')
            content = [e.get_text(strip=True) for e in content_elements if e.get_text(strip=True)]
            response = " ".join(content) if content else None
            logger.info(f"Extracted response content length: {len(response) if response else 0}")

            return query, response
        except Exception as e:
            logger.error(f"Failed to scrape Webex page at {self.url}: {e}")
            raise
