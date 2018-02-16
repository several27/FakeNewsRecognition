import logging
from typing import List

import pandas as pd
import peewee
from playhouse.shortcuts import model_to_dict
from tqdm import tqdm

from database import Page


def fetch_pages(last_id, batch_size):
    while True:
        pages = list(Page.select()
                     .where((Page.id > last_id) &
                            (peewee.fn.Length(Page.url) > 40) & (peewee.fn.Length(Page.content) > 40))
                     .order_by(Page.id.asc()).limit(batch_size))  # type: List[Page]

        logging.debug('Last id:%s' % (pages[-1].id if len(pages) > 0 else None))
        if len(pages) <= 0:
            return False

        for page in pages:
            yield page

            last_id = page.id


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    last_id = 0
    batch_size = 10 * 1000
    csv_columns = ['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title',
                   'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source']

    print('Processing')
    path_data = '/Volumes/ExternalSSD/FakeNewsRecognition/'
    logging.basicConfig(filename=path_data + 'data_full.log', level=logging.DEBUG)

    peewee_database_merged = peewee.SqliteDatabase(path_data + 'news_cleaned_2018_01_29+postgres+nytimes+webhose.db')
    Page._meta.database = peewee_database_merged

    with open(path_data + 'news_cleaned_2018_01_29+postgres+nytimes+webhose.csv', 'a') as out_news:
        df_pages = pd.DataFrame([], columns=csv_columns)
        df_pages.to_csv(out_news, header=True)

        with tqdm() as progress:
            pages_to_save = []
            for page in fetch_pages(last_id, batch_size):
                pages_to_save.append(page)

                if len(pages_to_save) >= batch_size:
                    df_pages = pd.DataFrame([model_to_dict(p) for p in pages_to_save], columns=csv_columns)
                    df_pages.to_csv(out_news, header=False)
                    pages_to_save = []

                progress.update()

            df_pages = pd.DataFrame([model_to_dict(p) for p in pages_to_save], columns=csv_columns)
            df_pages.to_csv(out_news, header=False)


if __name__ == '__main__':
    main()
