import os
from urllib.parse import urlparse

import scrapy
from scrapy.http import TextResponse
from scrapy.linkextractors import LinkExtractor
import newspaper
import pandas as pd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
path_fnr = os.environ['PATH_FNR']
path_7_fn = path_fnr + 'data/7_fake_news/'
path_websites = path_7_fn + 'websites_with_results.xlsx'


class NewsScraper(scrapy.Spider):
    name = 'news'

    def start_requests(self):
        df_websites = pd.read_excel(path_websites)
        websites_url = list(reversed(['http://' + u for u in df_websites[df_websites.result == 200].url.values]))

        for url in websites_url:
            yield scrapy.Request(url, self.parse)

    def parse(self, response: TextResponse):
        article = newspaper.Article(response.url)
        article.set_html(response.text)
        article.parse()

        yield {
            'title': article.title,
            'content': article.title,
            'published_at': article.publish_date,
            'authors': article.authors
        }

        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(response.url))

        for url in LinkExtractor(allow_domains=domain).extract_links(response):
            yield response.follow(url, callback=self.parse)
