from datamanagement.scraping.base import BaseScraper

class CommunityScraper(BaseScraper):
    def scrape(self):
        soup = self._get_soup()
        message_bodies = soup.select('.lia-message-body-content')
        query = message_bodies[0].get_text(strip=True) if message_bodies else None

        if not query or len(query.strip()) <= 1:
            title_elem = soup.select_one('title')
            query = title_elem.get_text(strip=True) if title_elem else "No title"

        accepted_solution = soup.select_one('.lia-message-body-accepted-solution-checkmark .lia-message-body-content')
        if accepted_solution:
            response = accepted_solution.get_text(strip=True)
        elif len(message_bodies) > 1:
            response = message_bodies[1].get_text(strip=True)
        else:
            response = None

        return query, response
