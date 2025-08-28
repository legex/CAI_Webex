"""
Module: link_collector.py

Uses Selenium to scrape community site thread links, handling dynamic pagination, solved/unread thread detection, 
and cookie banner acceptance. Collects unique thread URLs for further processing.

Logs progress and errors to a dedicated log file.
"""
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from core.logger import setup_logger

logger = setup_logger('link_collector', 'log/link_collector.log')

class LinkCollector:
    """
    Scrapes community forum pages to collect solved and unsolved thread URLs.

    Handles cookie acceptance, dynamic 'Load More' pagination,
    and extracts URLs from solved icons and unsolved unread threads.

    Attributes:
        website (str): Base URL of the community forum.
        source (str): Identifier for source context, e.g., 'community'.
        driver (webdriver.Chrome): Selenium WebDriver instance.
        wait (WebDriverWait): Selenium explicit wait object.
        urls (set): Set of unique thread URLs collected.
    """
    LOAD_MORE_SELECTOR = ".lia-link-navigation.load-more-button.lia-button.lia-button-primary"
    SOLVED_ICON_SELECTOR = "i.custom-thread-solved"
    COOKIE_CLOSE_SELECTOR = ".onetrust-close-btn-handler.onetrust-close-btn-ui.banner-close-button.ot-close-icon"
    UNSOLVED_ARTICLE_SELECTOR = "article.custom-message-tile.custom-thread-unread"

    def __init__(self, sources, website):
        """
        Initialize scraper with Selenium driver and parameters.

        Args:
            sources (str): Source identifier, e.g., 'community'.
            website (str): Community forum home URL.
        """
        self.website = website
        self.source = sources
        self.driver = webdriver.Chrome(options=self._default_options())
        self.wait = WebDriverWait(self.driver, 5)
        self.urls = set()
        logger.info(f"LinkCollector initialized for website: {website}")

    def _default_options(self):
        options = Options()
        # options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("start-maximized")
        return options

    def _accept_cookies(self):
        """Attempts to close cookie banner if present."""
        try:
            self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, self.COOKIE_CLOSE_SELECTOR))).click()
            logger.info("Cookies banner closed successfully.")
        except TimeoutException:
            logger.warning("No cookies banner found or timeout expired.")

    def _scroll_to_element(self, element):
        """Scrolls viewport to the specified element."""
        self.driver.execute_script("arguments[0].scrollIntoView();", element)
        time.sleep(0.4)

    def _click_load_more(self):
        """
        Clicks ‘Load More’ button if available and enabled.

        Returns:
            bool: True if clicked, False if button disabled (no more content).
        """
        try:
            load_more = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.LOAD_MORE_SELECTOR)))
            self._scroll_to_element(load_more)
            if 'disabled' in load_more.get_attribute("class"):
                logger.info("Load More button is disabled; no additional pages.")
                return False
            self.driver.execute_script("arguments[0].click();", load_more)
            time.sleep(0.4)
            logger.info("Load More button clicked.")
            return True
        except TimeoutException:
            logger.warning("Load More button not found or timeout expired.")
            return False

    def _extract_href_from_icon(self, icon):
        """
        Extracts and stores URL from solved thread icon element.

        Args:
            icon (WebElement): Selenium element containing solved icon.
        """
        try:
            h3 = icon.find_element(By.XPATH, "./ancestor::h3")
            a_tag = h3.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")
            if href:
                self.urls.add(href)
                logger.debug(f"Collected solved thread URL: {href}")
        except Exception as e:
            logger.error(f"Error extracting solved link: {e}")

    def _extract_href_from_unsolved_article(self, article):
        """
        Extracts and stores URL from an unsolved (unread) thread article element.

        Args:
            article (WebElement): Selenium element for unread thread.
        """
        try:
            h3 = article.find_element(By.TAG_NAME, "h3")
            a_tag = h3.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")
            if href:
                self.urls.add(href)
                logger.debug(f"Collected unsolved thread URL: {href}")
        except Exception as e:
            logger.error(f"Error extracting unsolved link: {e}")

    def scrape_website_community(self, max_pages=100):
        """
        Main method to scrape the community website for thread URLs.

        Args:
            max_pages (int, optional): Max number of times to click 'Load More'. Defaults to 100.

        Returns:
            list: List of unique thread URLs collected.
        """
        logger.info("Starting community site scrape...")
        self.driver.get(self.website)
        self._accept_cookies()

        pages_clicked = 0
        while pages_clicked < max_pages:
            try:
                if not self._click_load_more():
                    logger.info("No more content to load; ending pagination.")
                    break
                pages_clicked += 1
                logger.info(f"Total number of pages loaded: {pages_clicked}")
            except TimeoutException:
                logger.warning("Load More button not found or timed out.")
                break

        solved_icons = self.driver.find_elements(By.CSS_SELECTOR,
                                                 self.SOLVED_ICON_SELECTOR)
        for icon in solved_icons:
            self._extract_href_from_icon(icon)

        unsolved_articles = self.driver.find_elements(By.CSS_SELECTOR,
                                                      self.UNSOLVED_ARTICLE_SELECTOR)
        for article in unsolved_articles:
            self._extract_href_from_unsolved_article(article)
        logger.info(f"Collected {len(unsolved_articles)} unsolved (unread-thread) links.")

        logger.info(f"Total unique thread links collected: {len(self.urls)}")
        return list(self.urls)

    def close(self):
        """Closes Selenium WebDriver cleanly."""
        if self.driver:
            self.driver.quit()
            logger.info("Selenium WebDriver closed.")
