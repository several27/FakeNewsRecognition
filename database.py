from datetime import datetime
from typing import Set

from peewee import Model, DateTimeField, CharField, TextField, IntegerField, SqliteDatabase

raw_peewee_database = SqliteDatabase('data/7_opensources_co/news_spider_2018_01_29.db')
peewee_database = SqliteDatabase('data/7_opensources_co/news_cleaned_postgres.db')


class RawBaseModel(Model):
    class Meta:
        database = raw_peewee_database


class BaseModel(Model):
    class Meta:
        database = peewee_database


class ScrapedPage(RawBaseModel):
    batch = IntegerField(null=False)
    url = CharField(null=False)
    html = TextField(null=False)
    inserted_at = DateTimeField(null=False, default=datetime.now())
    updated_at = DateTimeField(null=False, default=datetime.now())

    # cache
    _cache_urls = None  # type: Set[str]

    @staticmethod
    def url_exists(url):
        if ScrapedPage._cache_urls is None:
            print('Getting all urls...')
            last_id = 0
            all_scraped = []
            while True:
                scraped = list(ScrapedPage
                               .select(ScrapedPage.id, ScrapedPage.url)
                               .where(ScrapedPage.id > last_id)
                               .order_by(ScrapedPage.id.asc())
                               .limit(100 * 1000))
                if len(scraped) <= 0:
                    break

                all_scraped += [sp.url for sp in scraped]
                last_id = scraped[-1].id

                print('Got %s with id: %s' % (len(all_scraped), last_id))

            ScrapedPage._cache_urls = set(all_scraped)
            print('Unique urls: %s' % len(ScrapedPage._cache_urls))

        return url in ScrapedPage._cache_urls

    class Meta:
        db_table = 'fnr_scraped_pages'

        indexes = (
            # Specify a unique multi-column index on from/to-user.
            (('batch', 'url'), True),
        )


class Page(BaseModel):
    scraped_page_id = IntegerField(null=False)
    batch = IntegerField(null=False)
    domain = CharField(null=False)
    type = CharField(null=True)
    url = CharField(null=False)
    source = CharField(null=True)
    content = TextField(null=False)
    scraped_at = DateTimeField(null=False)
    inserted_at = DateTimeField(null=False, default=datetime.now())
    updated_at = DateTimeField(null=False, default=datetime.now())

    # additional fields
    title = TextField(null=True)
    authors = TextField(null=True)
    keywords = TextField(null=True)
    meta_keywords = TextField(null=True)
    meta_description = TextField(null=True)
    tags = TextField(null=True)
    summary = TextField(null=True)

    # cache
    _cache_urls = None  # type: Set[str]

    class Meta:
        db_table = 'fnr_pages'


peewee_database.create_tables([Page], safe=True)
