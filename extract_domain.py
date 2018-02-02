from typing import List
from urllib.parse import urlsplit

import pandas as pd
from tqdm import tqdm

from database import Page, peewee_database


def fetch_pages(last_id, batch_size):
    while True:
        print('Fetching batch of', batch_size)
        pages = list(Page.select(Page.id, Page.url) \
                     .where((Page.batch == 2) & (Page.id > last_id) &
                            (Page.id <= (last_id + batch_size))) \
                     .order_by(Page.id.asc()))  # type: List[Page]

        print('Fetched', len(pages))
        if len(pages) <= 0:
            return

        for page in pages:
            yield page

            last_id = page.id


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    last_id = 402402
    batch_size = 100 * 1000
    urls_domains_not_found = []

    df_websites = pd.read_excel('data/7_opensources_co/websites_with_results.xlsx')
    domains = [u for u in df_websites.url.values]

    print('Processing')
    with open('data/7_opensources_co/news_cleaned_2018_01_29_missing_domains.csv', 'w') as out_missing_domains:
        with tqdm() as progress:
            pages_to_save = []
            for page in fetch_pages(last_id, batch_size):
                domain = None
                for d in domains:
                    if d in page.url:
                        domain = d

                if domain is None:
                    urls_domains_not_found.append(page.url)
                    out_missing_domains.write(page.url + '\n')

                    domain = urlsplit(page.url).netloc

                page.domain = domain
                pages_to_save.append(page)

                if len(pages_to_save) > 1000:
                    print('Last id:', pages_to_save[-1].id)
                    with peewee_database.atomic():
                        for _page in pages_to_save:
                            _page.save()

                        pages_to_save = []

                progress.update()

            with peewee_database.atomic():
                for _page in pages_to_save:
                    _page.save()

    print('Urls without our domains?!:', len(urls_domains_not_found))


if __name__ == '__main__':
    main()
