# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


import os
from datetime import datetime
from urllib.parse import urlsplit

import ujson
from peewee import Model
from playhouse.postgres_ext import PostgresqlExtDatabase, DateTimeField, CharField, TextField, IntegerField
from scrapy.exceptions import DropItem

from .spiders import NewsScraper

peewee_database = PostgresqlExtDatabase(os.environ['RESEARCHABLY_DB_NAME'], **{
    'host': os.environ    ['RESEARCHABLY_DB_HOST'],
    'user': os.environ    ['RESEARCHABLY_DB_USER'],
    'password': os.environ['RESEARCHABLY_DB_PASSWORD'],
    'register_hstore':    False
})


class BaseModel(Model):
    class Meta:
        database = peewee_database


class ScrapedPage(BaseModel):
    batch = IntegerField(null=False)
    url = CharField(null=False)
    html = TextField(null=False)
    inserted_at = DateTimeField(null=False, default=datetime.now())
    updated_at = DateTimeField(null=False, default=datetime.now())

    class Meta:
        db_table = 'fnr_scraped_pages'


class NewsSpiderDropPipeline:
    def __init__(self):
        self.netlocs = set()

    def open_spider(self, spider: NewsScraper):
        for u in spider.websites_url:
            self.netlocs.add(urlsplit(u).netloc.lower())

    def process_item(self, item, spider):
        url = item['url'].replace('www.', '').lower()
        if urlsplit(url).netloc not in self.netlocs:
            raise DropItem()

        return item


class NewsSpiderPersistencePipeline(object):
    def __init__(self):
        self.items = []
        with open('../data/7_opensources_co/scraped_pages_urls.json', 'r') as _in:
            self.scraped_urls = set(ujson.load(_in))

    def process_item(self, item, scraper):
        if item['url'] in self.scraped_urls:
            return item

        self.items.append({
            'batch': 3,
            'url': item['url'],
            'html': item['html'].replace('\x00', '')
        })

        if len(self.items) >= 100:
            self.process_bulk()

        return item

    def close_spider(self, spider):
        self.process_bulk()

    def process_bulk(self):
        if peewee_database.is_closed():
            peewee_database.connect()

        with peewee_database.atomic():
            ScrapedPage.insert_many(self.items).execute()

        self.items = []
