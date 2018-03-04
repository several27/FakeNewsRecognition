import os

import ujson
from tqdm import tqdm
from peewee import SqliteDatabase, fn
from playhouse.shortcuts import model_to_dict

from database import Page

path_data = '/Volumes/ExternalSSD/FakeNewsRecognition/'
path_source = path_data + 'news_cleaned_2018_01_29+postgres+nytimes+webhose.db'
path_destination = path_data + 'news_cleaned_2018_03_03.db'


def insert_bulk(pages, db):
    Page._meta.database = db
    db.connect(reuse_if_open=True)

    with db.atomic():
        Page.insert_many(pages).execute()


def fetch_pages(last_id, batch_size, db):
    while True:
        Page._meta.database = db
        db.connect(reuse_if_open=True)

        pages = list(Page.select()
                     .where((Page.id > last_id) & (Page.id <= (last_id + batch_size)))
                     .order_by(Page.id.asc()))

        if len(pages) <= 0:
            return

        for page in pages:
            yield page

            last_id = page.id


def main():
    peewee_database_source = SqliteDatabase(path_source)
    peewee_database_destination = SqliteDatabase(path_destination)

    Page._meta.database = peewee_database_destination
    Page.create_table()

    counter = {
        'content_skipped': []
    }

    unique_hashes = {
        'title_url': set(),
        'title_content': set(),
        'url': set(),
        'content': set()
    }

    with tqdm() as progress:
        batch = []
        for page in fetch_pages(0, 1000, peewee_database_source):
            progress.update()

            content_hash = page.content.__hash__()

            if content_hash in unique_hashes['content']:
                counter['content_skipped'].append(page.id)
                continue

            unique_hashes['content'].add(content_hash)
            batch.append(model_to_dict(page))

            if len(batch) > 50:
                insert_bulk(batch, peewee_database_destination)
                batch = []


if __name__ == '__main__':
    main()
