import os

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.domains = []
        self.websites_url = []

    def start_requests(self):
        df_websites = pd.read_excel(path_websites)
        self.domains = [u for u in df_websites[df_websites.result == 200].url.values]
        self.websites_url = ['http://' + u for u in self.domains]

        for url in self.websites_url:
            yield scrapy.Request(url, self.parse)

    def parse(self, response: TextResponse):
        # article = newspaper.Article(response.url)
        # article.set_html(response.text)
        # article.parse()

        yield {
            # 'title': article.title,
            # 'content': article.text,
            # 'published_at': article.publish_date,
            # 'authors': article.authors,
            'url': response.url,
            'html': response.text
        }

        for url in LxmlLinkExtractor(allow_domains=self.domains).extract_links(response):
            yield response.follow(url, callback=self.parse)

        # for url in response.css('a::attr(href)').extract():
        #     if len(url) > 1 and (urlsplit(response.url).netloc == urlsplit(url).netloc or
        #                          (url[0] == '/' and url[1] != '/')):

        # for url in LinkExtractor(allow_domains=domain, tags=('a',), attrs=('href',)).extract_links(response):
        #     print('---', url)
        #     yield response.follow(url, callback=self.parse)


