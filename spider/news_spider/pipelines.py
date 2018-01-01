# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


import os
import random
from datetime import datetime
from urllib.parse import urlsplit

import ujson

import gevent.monkey
from psycopg2.extras import RealDictCursor
from peewee import Model
from playhouse.postgres_ext import PostgresqlExtDatabase, DateTimeField, CharField, TextField, IntegerField
from scrapy.exceptions import DropItem
from tqdm import tqdm

from .spiders import NewsScraper

peewee_database = PostgresqlExtDatabase(os.environ['RESEARCHABLY_DB_NAME'], **{
    'host': os.environ['RESEARCHABLY_DB_HOST'],
    'user': os.environ['RESEARCHABLY_DB_USER'],
    'password': os.environ['RESEARCHABLY_DB_PASSWORD'],
    'register_hstore': False
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


def big_select_query(query, batch_size=100 * 1000):
    peewee_database.connect()

    cursor = peewee_database._local.conn.cursor(cursor_factory=RealDictCursor)
    cursor_name = 'task_big_query' + str(random.randint(1, 1000 * 1000))
    cursor.execute('begin; declare ' + cursor_name + ' cursor for ' + query.sql()[0], query.sql()[1])

    while True:
        cursor.execute('fetch ' + str(batch_size) + ' from ' + cursor_name + ' ;')
        batch_results = cursor.fetchall()

        if len(batch_results) <= 0:
            break

        for row in batch_results:
            yield row


class NewsSpiderDropPipeline:
    def __init__(self):
        self.netlocs = set()
        self.existing_urls = set()

    def open_spider(self, spider: NewsScraper):
        for u in spider.websites_url:
            self.netlocs.add(urlsplit(u).netloc.lower())

        self.existing_urls = self.query_for_scraped_urls()

    def process_item(self, item, spider):
        url = item['url'].replace('www.', '').lower()
        if urlsplit(url).netloc not in self.netlocs or item['url'][:50] in self.existing_urls:
            item['html'] = None
            raise DropItem()

        self.existing_urls.add(item['url'])

        return item

    @staticmethod
    def query_for_scraped_urls():
        print('Querying already scraped urls...')

        urls = set()
        gevent.monkey.patch_all(thread=False)
        for page in tqdm(big_select_query(ScrapedPage.select(ScrapedPage.url)
                                                  .where((ScrapedPage.batch == 2) | (ScrapedPage.batch == 3) |
                                                         (ScrapedPage.batch == 4) | (ScrapedPage.batch == 5)))):
            urls.add(page['url'][:50])

        print('Querying already scraped urls... Finished!')

        return urls


class NewsSpiderPersistencePipeline(object):
    def __init__(self):
        self.items = []

    def process_item(self, item, scraper):
        self.items.append({
            'batch': 5,
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

        try:
            with peewee_database.atomic():
                ScrapedPage.insert_many(self.items).execute()
        except Exception as e:
            print(e)
            print('Retrying connection')
            peewee_database.connect()
            self.process_bulk()
            return

        self.items = []
