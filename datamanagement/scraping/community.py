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
        else:
            if len(message_bodies) > 1:
                next_bodies = [body.get_text(strip=True) for body in message_bodies[1:6]]
                response = "\n\n".join(next_bodies) if next_bodies else None
            else:
                response = None

        return query, response
