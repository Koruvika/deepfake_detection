"""
    Run crawl image from flickr.com website
"""
import os
import time
from crawl import crawl

crawl_url = "https://www.flickr.com/groups/human_faces/pool/page"
start_pages = 1
end_pages = 55  # Number of downloaded pages
output_directory = "raw_images"  # Raw image folder

# Create directory if not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

start_time = time.time()
# Run crawl image function
print(f"Crawling image in page {start_pages} to {end_pages}...")
crawl(crawl_url, output_directory, start_pages, end_pages)

end_time = time.time()
print(f"Total download time: {(end_time - start_time)/60:.2f} (minutes)")
