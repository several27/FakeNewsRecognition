# coding=utf-8

import sys
import multiprocessing
from typing import Type
from urllib.parse import urlsplit

import peewee
import newspaper
import pandas as pd
from tqdm import tqdm

from preprocessing.database import Page, ScrapedPage


def parse_article(scraped_page: ScrapedPage):
    if len(scraped_page.html) <= 0 or len(scraped_page.url) <= 0:
        return None

    try:
        article = newspaper.Article(scraped_page.url, fetch_images=False)
        article.set_html(scraped_page.html)
        article.parse()
    except Exception:
        print('Something went wrong parsing id:', scraped_page.id, 'url:', scraped_page.url)
        return None

    if len(article.text) <= 40:
        return None

    return {
        'batch': 2,
        'scraped_page_id': scraped_page.id,
        'url': scraped_page.url,
        'scraped_at': scraped_page.updated_at,
        'title': article.title,
        'content': article.text,
        'authors': ', '.join(article.authors),
        'keywords': ', '.join(article.keywords),
        'meta_keywords': article.meta_keywords,
        'meta_description': article.meta_description,
        'tags': ', '.join(article.tags),
        'summary': article.summary
    }


def fetch_pages(last_id, batch_size, db, page_model: Type[peewee.Model] = Page, conditions=None, filter_=None):
    while True:
        page_model._meta.database = db
        db.connect(reuse_if_open=True)

        pages = list(page_model.select()
                     .where((page_model.id > last_id) & (peewee.fn.Length(page_model.url) > 40) & conditions)
                     .order_by(page_model.id.asc()).limit(batch_size))

        if len(pages) <= 0:
            return

        for page in pages:
            if filter_ is None or filter_(page):
                yield page

            last_id = page.id


def main():
    path_data = 'data/ssd/'
    peewee_database_destination = peewee.SqliteDatabase(path_data + 'news_cleaned_2018_03_26.db')
    peewee_database_source = peewee.SqliteDatabase(path_data + 'scraping_2018_03_10/news_spider.db')

    path_missing_domains = path_data + '2018_02_13_missing_domains.csv'
    path_websites = path_data + 'websites_with_results.xlsx'

    batch_size = 10 * 1000

    df_websites = pd.read_excel(path_websites)
    domains = [u for u in df_websites.url.values]

    domain_type = {}
    websites_url = df_websites.url.values
    websites_type = df_websites.type.values
    for i, url in enumerate(websites_url):
        domain_type[url] = websites_type[i]

    urls_domains_not_found = []

    unique_hashes = {
        'url': set(),
        'content': set()
    }

    print('Fetching destination URL hashes')
    with tqdm() as progress:
        for page in fetch_pages(0, batch_size, peewee_database_destination, Page,
                                conditions=(peewee.fn.Length(Page.url) > 0)):
            unique_hashes['url'].add(hash(page.url))
            unique_hashes['content'].add(hash(page.content))
            progress.update()

    print('Found %s unique urls and %s unique contents' % (len(unique_hashes['url']),
                                                           len(unique_hashes['content'])))

    print('Piping data from source to destination')
    with open(path_missing_domains, 'w') as out_missing_domains:
        with tqdm() as progress:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

                def fetch_pages_filter(page_: ScrapedPage):
                    progress.update()

                    if hash(page_.url) in unique_hashes['url']:
                        return False

                    unique_hashes['url'].add(hash(page_.url))
                    return True

                pages_parsed = []
                fetch_pages_ = fetch_pages(0, batch_size, peewee_database_source, ScrapedPage,
                                           conditions=(peewee.fn.Length(ScrapedPage.html) > 0),
                                           filter_=fetch_pages_filter)
                for page in pool.imap_unordered(parse_article, fetch_pages_):
                    if page is None:
                        continue

                    domain = None
                    for d in domains:
                        if d in page['url']:
                            domain = d

                    if domain is None:
                        urls_domains_not_found.append(page['url'])
                        out_missing_domains.write(page['url'] + '\n')

                    if hash(page['content']) in unique_hashes['content']:
                        continue

                    page['domain'] = urlsplit(page['url']).netloc
                    page['type'] = domain_type[page['domain']] if page['domain'] in domain_type else None

                    unique_hashes['content'].add(hash(page['content']))

                    pages_parsed.append(page)

                    if len(pages_parsed) > 50:
                        progress.write('Inserting cleaned articles to DB')
                        with peewee_database_destination.atomic():
                            Page._meta.database = peewee_database_destination
                            Page.insert_many(pages_parsed).execute()

                        progress.write('Inserted %s articles' % len(pages_parsed))
                        pages_parsed = []

    # TODO: still needs to fill up the source field

if __name__ == '__main__':
    main()
