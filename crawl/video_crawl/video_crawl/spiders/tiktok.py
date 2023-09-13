import scrapy
from urllib.parse import urlencode
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

SEARCH_LINK = "https://www.tiktok.com/search"

class TiktokCrawler(scrapy.Spider):
    name = "tiktok"
    allowed_domains = ["tiktok.com"]

    query = "Elon Musk Interview"
    count = 20

    def request(self, page_number: int):
        query = {
            "q": self.query,
        }

        return [scrapy.Request(
            url=SEARCH_LINK + "?" + urlencode(query=query),
            callback=self.parse,
        )]

    def start_requests(self):
        return self.request(page_number=0)

    def parse(self, response):
        print(response.url)
        # Initialize web driver
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("start-maximized")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        driver = webdriver.Chrome(options=chrome_options, executable_path=r'crawl/webdriver/chromedriver_mac64')

        driver.get(response.url)

        driver.execute_script("return arguments[0].scrollIntoView(true);", WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, "//h1//span[text()='Distribution']"))))
        elements = WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//h1//span[text()='Distribution']//following::div[1]/*[name()='svg']//*[name()='g']//*[name()='g' and @class='paper']//*[name()='circle']")))
        for element in elements:
            ActionChains(driver).move_to_element(element).perform()

        driver.close()