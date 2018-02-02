from typing import List
from urllib.parse import urlsplit

import pandas as pd
import peewee
from playhouse.shortcuts import model_to_dict
from tqdm import tqdm

from database import Page, peewee_database


def fetch_pages(last_id, batch_size):
    while True:
        print('Fetching batch of', batch_size)
        pages = list(Page.select()\
                     .where((Page.batch == 2) & (Page.id > last_id) &
                            (Page.id <= (last_id + batch_size)) &
                            (peewee.fn.Length(Page.url) > 40) & (peewee.fn.Length(Page.content) > 40)) \
                     .order_by(Page.id.asc()))  # type: List[Page]

        print('Fetched', len(pages))
        if len(pages) <= 0:
            return

        for page in pages:
            yield page

            last_id = page.id


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    last_id = 0
    batch_size = 1000
    csv_columns = ['id', 'domain', 'type', 'url', 'content', 'scraped_at', 'inserted_at', 'updated_at', 'title',
                   'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary']

    print('Processing')
    with open('data/7_opensources_co/news_cleaned_2018_01_29.csv', 'a') as out_news:
        df_pages = pd.DataFrame([], columns=csv_columns)
        df_pages.to_csv(out_news, header=True)

        with tqdm() as progress:
            pages_to_save = []
            for page in fetch_pages(last_id, batch_size):
                pages_to_save.append(page)

                if len(pages_to_save) >= 1000:
                    df_pages = pd.DataFrame([model_to_dict(p) for p in pages_to_save], columns=csv_columns)
                    df_pages.to_csv(out_news, header=False)
                    pages_to_save = []

                progress.update()

            df_pages = pd.DataFrame([model_to_dict(p) for p in pages_to_save], columns=csv_columns)
            df_pages.to_csv(out_news, header=False)


if __name__ == '__main__':
    main()
