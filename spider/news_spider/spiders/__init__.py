import os
from urllib.parse import urlsplit

import scrapy
from scrapy.http import TextResponse
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
import newspaper
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
path_fnr = os.environ['PATH_FNR']
path_7_fn = path_fnr + 'data/7_opensources_co/'
path_websites = path_7_fn + 'websites_with_results.xlsx'
path_logger = path_7_fn + 'news_spider.log'


class NewsScraper(scrapy.Spider):
    name = 'news'

    def __init__(self, websites_start=0, websites_end=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        df_websites = pd.read_excel(path_websites)
        if websites_end is None:
            websites_end = len(df_websites)

        df_websites = df_websites[int(websites_start):int(websites_end)]
        self.domains = [u for u in df_websites[df_websites.result == 200].url.values]
        self.websites_url = ['http://' + u for u in self.domains]

    def start_requests(self):
        for url in self.websites_url:
            yield scrapy.Request(url, self.parse)

    def parse(self, response: TextResponse):
        # article = newspaper.Article(response.url)
        # article.set_html(response.text)
        # article.parse()

        if response.status != 200:
            return

        yield {
            # 'title': article.title,
            # 'content': article.text,
            # 'published_at': article.publish_date,
            # 'authors': article.authors,
            'url': response.url,
            'html': response.text
        }

        for link in LxmlLinkExtractor(allow_domains=self.domains).extract_links(response):
            split = urlsplit(link.url)
            scheme = (split.scheme + '://') if len(split.scheme) > 1 else ''
            if len(scheme) < 1 and len(split.netloc) > 1 and split.netloc[0] != '/':
                scheme = '//'

            url = scheme + split.netloc + split.path

            yield response.follow(url, callback=self.parse)
