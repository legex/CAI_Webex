from abc import ABC, abstractmethod
from datamanagement.scraping.url_access import URLAccess

class BaseScraper(ABC):
    def __init__(self, source, url):
        self.source = source
        self.url = url

    def _get_soup(self):
        return URLAccess(self.source, self.url).content()

    @abstractmethod
    def scrape(self):
        pass
