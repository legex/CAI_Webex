import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

logger = logging.getLogger(__name__)

class LinkCollector:
    LOAD_MORE_SELECTOR = ".lia-link-navigation.load-more-button.lia-button.lia-button-primary"
    SOLVED_ICON_SELECTOR = "i.custom-thread-solved"
    COOKIE_CLOSE_SELECTOR = ".onetrust-close-btn-handler.onetrust-close-btn-ui.banner-close-button.ot-close-icon"
    UNSOLVED_ARTICLE_SELECTOR = "article.custom-message-tile.custom-thread-unread"

    def __init__(self, sources, website):
        self.website = website
        self.source = sources
        self.driver = webdriver.Chrome(options=self._default_options())
        self.wait = WebDriverWait(self.driver, 5)
        self.urls = set()

    def _default_options(self):
        options = Options()
        # options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("start-maximized")
        return options

    def _accept_cookies(self):
        try:
            self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, self.COOKIE_CLOSE_SELECTOR))).click()
            logger.info("Cookies banner closed")
        except TimeoutException:
            logger.warning("No cookies banner found.")

    def _scroll_to_element(self, element):
        self.driver.execute_script("arguments[0].scrollIntoView();", element)
        time.sleep(0.4)

    def _click_load_more(self):
        load_more = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.LOAD_MORE_SELECTOR)))
        self._scroll_to_element(load_more)
        if 'disabled' in load_more.get_attribute("class"):
            return False
        self.driver.execute_script("arguments[0].click();", load_more)
        time.sleep(0.4)
        return True

    def _extract_href_from_icon(self, icon):
        try:
            h3 = icon.find_element(By.XPATH, "./ancestor::h3")
            a_tag = h3.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")
            if href:
                self.urls.add(href)
        except Exception as e:
            logger.error(f"Error extracting solved link: {e}")

    def _extract_href_from_unsolved_article(self, article):
        """Get thread URL given an unsolved (unread) article element."""
        try:
            h3 = article.find_element(By.TAG_NAME, "h3")
            a_tag = h3.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")
            if href:
                self.urls.add(href)
        except Exception as e:
            logger.error(f"Error extracting unsolved link: {e}")

    def scrape_website_community(self, max_pages=100):
        logger.info("Starting community site scrape...")
        self.driver.get(self.website)
        self._accept_cookies()

        pages_clicked = 0
        while pages_clicked < max_pages:
            try:
                if not self._click_load_more():
                    logger.info("No more items to load.")
                    break
                pages_clicked += 1
                print(f"Total number of pages: {pages_clicked}")
            except TimeoutException:
                logger.warning("Load More button not found or timed out.")
                break

        solved_icons = self.driver.find_elements(By.CSS_SELECTOR, self.SOLVED_ICON_SELECTOR)
        for icon in solved_icons:
            self._extract_href_from_icon(icon)

        unsolved_articles = self.driver.find_elements(By.CSS_SELECTOR, self.UNSOLVED_ARTICLE_SELECTOR)
        for article in unsolved_articles:
            self._extract_href_from_unsolved_article(article)
        logger.info(f"Collected {len(unsolved_articles)} unsolved (unread-thread) links.")

        logger.info(f"Total unique thread links collected: {len(self.urls)}")
        return list(self.urls)

    def close(self):
        if self.driver:
            self.driver.quit()
