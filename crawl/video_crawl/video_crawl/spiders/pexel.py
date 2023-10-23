from scrapy import Spider, Request
from urllib.parse import urlencode
import json
from ..items import VideoCrawlItem

SEARCH_URL = "https://api.pexels.com/videos/search"

# Your Pexel API KEY here
API_KEY = "a9aepe75HdUG3pp4l1ev4dCuctijyaROdktSHLf1Os6EwXgXQ3rJheRZ"


class PerxelCrawler(Spider):
    name = "pexel"

    query = "Short representation"

    count = 100
    per_page = 30

    def request(self, page_number: int):
        query = {"query": self.query, "page": 1, "per_page": self.per_page}

        HEADERS = {"Authorization": API_KEY}
        return Request(
            url=SEARCH_URL + "?" + urlencode(query),
            callback=self.parse,
            headers=HEADERS,
            meta={"page": page_number},
        )

    def start_requests(self):
        yield self.request(page_number=1)

    def parse(self, response):
        resp = json.loads(response.body)
        for video in resp["videos"]:
            video_file = video["video_files"][0]
            yield VideoCrawlItem(
                id=video_file["id"],
                file_urls=[video_file["link"]],
            )

        if (int(resp["page"]) + 1) * int(self.per_page) < int(self.count):
            yield self.request(page_number=response["page"] + 1)
