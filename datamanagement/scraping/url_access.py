"""
Module: url_access.py

Defines URLAccess class to fetch and parse HTML content from specified sources ('community' or 'webex').
Parses links matching source-specific patterns and returns normalized absolute URLs.

Uses sslbypass module for legacy session handling to bypass SSL issues if needed.
"""
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from datamanagement.core.logger import setup_logger
from datamanagement.scraping import sslbypass

logger = setup_logger('url_access', 'datamanagement/log/url_access.log')

class URLAccess:
    """
    Accesses and parses URLs within the defined source namespaces.
    
    Attributes:
        source (str): 'community' or 'webex', representing supported source categories.
        url (str): The base URL to retrieve and parse.
        links (list): Cached list of parsed, filtered, absolute URLs.
        soup (BeautifulSoup): Parsed HTML content of the page.
    """
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
        'community': r'^/c/en/us',
        'webex': r'^/en-us/(article/|[\w-]+$)'
    }

    def __init__(self, source: str):
        """
        Initialize URLAccess with a source and the target URL to parse.

        Args:
            source (str): Source identifier ('community' or 'webex').
            url (str): Full URL to fetch and parse.

        Raises:
            ValueError: If the source is unsupported.
            ConnectionError: If fetching the URL fails or returns non-HTML.
        """
        if source not in self.base_urls:
            logger.error(f"Unsupported source specified: {source}")
            raise ValueError(f"Unsupported source: {source}")
        self.source = source
        self.links = []
        logger.info(f"Initialized URLAccess for {source}")

    def _fetch_html(self, url: str):
        """
        Uses sslbypass to get a session and fetch HTML content from self.url.

        Returns:
            BeautifulSoup: Parsed HTML using lxml or html.parser.

        Raises:
            ConnectionError: For HTTP failures or non-HTML content.
        """
        try:
            session = sslbypass.get_legacy_session()
            response = session.get(url)
            if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                logger.info(f"Successfully fetched HTML content from {url}")
                return BeautifulSoup(response.text, 'html.parser')
            else:
                msg = f"Failed retrieving HTML from {url} " \
                      f"(Status code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')})"
                logger.error(msg)
                raise ConnectionError(msg)
        except Exception as exc:
            logger.error(f"Exception during HTTP GET for {url}: {exc}")
            raise

    def _soup(self, url):
        return self._fetch_html(url)

    def linksparsed(self, url) -> list:
        """
        Parse and extract filtered absolute links from the fetched HTML content.

        Returns:
            list: Unique list of filtered absolute URLs related to the source context.
        """
        soup = self._fetch_html(url)
        try:
            raw_links = []
            base_url = self.base_urls[self.source][0]
            pattern = self.href_patterns[self.source]
            for a_tag in soup.find_all('a', href=re.compile(pattern)):
                href = a_tag.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    raw_links.append(full_url)

            # Deduplicate and remove base URLs themselves
            all_links = list(set(filter(lambda link: link not in self.base_urls[self.source], raw_links)))

            self.links = all_links
            logger.info(f"Parsed {len(all_links)} unique links from {url}")
            return all_links
        except Exception as exc:
            logger.error(f"Failed during parsing links from {url}: {exc}")
            raise

    def content(self, url) -> BeautifulSoup:
        """
        Get the parsed BeautifulSoup HTML content.

        Returns:
            BeautifulSoup: Parsed HTML content object.
        """
        return self._fetch_html(url)
