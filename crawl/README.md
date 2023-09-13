# Video Crawler

Short video crawler based on [scrapy](https://scrapy.org/) with user's query

## Usage

Requirements :

- itemadapter==0.3.0
- Scrapy==2.8.0
- selenium==4.12.0
- Adding Pexel API key to [pexel.py](video_crawl/video_crawl/spiders/pexel.py) file

### How to run:

- Example :

```bash
cd crawl/video_crawl

scrapy crawl pexel -a count=30 -a query='Elon Mush Short Interview'
```

- Option:

```
    count : Number of videos to crawl
    query : Query for searching video
```

### Result:

- The output will be stored in /videos folder with the name is the id of the video