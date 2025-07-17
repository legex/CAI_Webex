from datamanagement.scraping.base import BaseScraper

class WebexScraper(BaseScraper):
    def scrape(self):
        soup = self._get_soup()
        title = soup.select_one("title")
        query = title.get_text(strip=True).replace(" - Webex Help Center", "") if title else None

        content = [e.get_text(strip=True) for e in soup.select('p, li') if e.get_text(strip=True)]
        response = " ".join(content) if content else None

        return query, response
