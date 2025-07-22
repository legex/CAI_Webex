"""
Module: community_scraper.py

Implements a scraper for community forum pages by extracting the user query and accepted solution or follow-up messages.
Depends on BaseScraper for common scraping utilities.
"""
from datamanagement.scraping.base import BaseScraper
from datamanagement.core.logger import setup_logger

logger = setup_logger('community_scraper', 'datamanagement/log/community_scraper.log')

class CommunityScraper(BaseScraper):
    """
    Scraper to extract question and accepted answer (or follow-up messages) from community forum pages.

    Extracts:
        - Query: The first message body text or page title fallback.
        - Response: Accepted solution message if present; otherwise concatenates next 5 message bodies.
    """
    def scrape(self):
        """
        Scrapes the community page to retrieve the main query and corresponding response.

        Returns:
            tuple(str or None, str or None): A tuple of (query_text, response_text)
        """
        try:
            soup = self._get_soup()
            message_bodies = soup.select('.lia-message-body-content')
            query = message_bodies[0].get_text(strip=True) if message_bodies else None

            if not query or len(query.strip()) <= 1:
                title_elem = soup.select_one('title')
                query = title_elem.get_text(strip=True) if title_elem else "No title"
            
            logger.info(f"Extracted query text (length {len(query) if query else 0}) from {self.url}")

            accepted_solution = soup.select_one('.lia-message-body-accepted-solution-checkmark .lia-message-body-content')
            if accepted_solution:
                response = accepted_solution.get_text(strip=True)
                logger.info("Accepted solution found and extracted.")
            else:
                if len(message_bodies) > 1:
                    next_bodies = [body.get_text(strip=True) for body in message_bodies[1:6]]
                    response = "\n\n".join(next_bodies) if next_bodies else None
                    logger.info(f"No accepted solution; extracted {len(next_bodies)} follow-up messages.")
                else:
                    response = None
                    logger.warning("No accepted solution and no follow-up messages found.")

            return query, response
        except Exception as e:
            logger.error(f"Failed to scrape community page at {self.url}: {e}")
            raise
