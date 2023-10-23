"""
    Crawl image function from flickr.com website
"""

import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from tqdm import tqdm

# Initialize headless for ChromeDriver (not load website)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Run in headless mode (hidden)

def crawl(crawl_url, output_directory, start_pages, end_pages):
    """
        Crawl image function
    :return:
    """

    total_downloaded_images = 0     # Total downloaded image

    # Loop in all page
    for page_number in range(start_pages, end_pages + 1):
        # URL of Flickr page
        page_url = f"{crawl_url}{page_number}"

        # Use Chrome we need ChromeDriver
        ser = Service("C:/webdrivers/chromedriver.exe")
        driver = webdriver.Chrome(service=ser, options=chrome_options)

        # Open web in browser
        driver.get(page_url)

        scroll_count = 8  # Number of page scroll
        for _ in range(scroll_count):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)  # Waiting load page time

        # Get HTML after scrolling
        page_source = driver.page_source
        driver.quit()

        # Get content HTML page
        soup = BeautifulSoup(page_source, "html.parser")

        # Find all img element in HTML
        image_divs = soup.find_all("div", class_="photo-list-photo-container")

        # Total image need to download
        total_files = len(image_divs)

        # Initialize tqdm process bar for this page
        page_progress_bar = tqdm(total=total_files, unit="image", desc=f"Page {page_number} Progress")

        # Download and save image with progress bar for this page
        for i, image_div in enumerate(image_divs):
            img_tag = image_div.find("img")
            img_url = img_tag.get("src")
            if not img_url.startswith("http:") and not img_url.startswith("https:"):
                img_url = "http:" + img_url
            img_url = '/'.join(img_url.split('/'))

            if img_url:
                try:
                    # Download image
                    response = requests.get(img_url, stream=True)
                    if response.status_code == 200:
                        image_name = f"image_{total_downloaded_images + 1}.jpg"     # Image name
                        image_path = os.path.join(output_directory, image_name)

                        with open(image_path, "wb") as file:
                            for chunk in response.iter_content(1024):
                                file.write(chunk)

                        # Update tqdm progress bar
                        page_progress_bar.update(1)
                        # Update total downloaded image
                        total_downloaded_images += 1
                    else:
                        print(f"Failed to download: {img_url}")
                except Exception as e:
                    print(f"Error: {str(e)}")

        # Close tqdm progress bar for this page
        page_progress_bar.close()

        print(f"Page {page_number} is complete. Downloaded {total_downloaded_images} images in total.")
