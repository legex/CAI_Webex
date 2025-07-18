from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from datamanagement.scraping import sslbypass

class URLAccess:
    base_urls = {
        'community': [
            'https://community.cisco.com/t5/webex-meetings-and-webex-app/bd-p/disc-webex',
            'https://community.cisco.com/t5/webex-community/ct-p/webex-user'
        ],
        'webex': [
            'https://help.webex.com/en-us',
            'https://help.webex.com/contact'
        ]
    }

    href_patterns = {
        'community': r'^/t5/webex-',
        'webex': r'^/en-us/(article/|[\w-]+$)'
    }

    def __init__(self, source: str, url: str):
        if source not in self.base_urls:
            raise ValueError(f"Unsupported source: {source}")
        self.source = source
        self.url = url
        self.links = []
        self.soup = self._fetch_html()

    def _fetch_html(self):
        session = sslbypass.get_legacy_session()
        response = session.get(self.url)
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            return BeautifulSoup(response.text, 'html.parser')
        else:
            raise ConnectionError(f"Failed to retrieve HTML from {self.url} (Status: {response.status_code})")

    def linksparsed(self):
        raw_links = []
        base_url = self.base_urls[self.source][0]
        for a_tag in self.soup.find_all('a', href=re.compile(self.href_patterns[self.source])):
            href = a_tag.get('href')
            if href:
                full_url = urljoin(base_url, href)
                raw_links.append(full_url)

        all_links = list(set(filter(lambda link: link not in self.base_urls[self.source], raw_links)))
        self.links = all_links
        return self.links

    def content(self):
        return self.soup
