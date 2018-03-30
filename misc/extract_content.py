from typing import List
from urllib.parse import urlsplit

import multiprocessing
import multiprocessing.pool

import ujson
import pandas as pd
from tqdm import tqdm
import newspaper

from .database import Page, ScrapedPage, peewee_database


def parse_article(scraped_page: ScrapedPage):
    if len(scraped_page.html) <= 0:
        print('Missing url for id', scraped_page.id)
        return None

    try:
        article = newspaper.Article(scraped_page.url, fetch_images=False)
        article.set_html(scraped_page.html)
        article.parse()
    except Exception:
        print('Something went wrong parsing id:', scraped_page.id, 'url:', scraped_page.url)
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


def fetch_pages(last_id, batch_size):
    while True:
        print('Fetching batch of', batch_size)
        scraped_pages = list(ScrapedPage.select()\
                             .where((ScrapedPage.batch == 2) & (ScrapedPage.id > last_id) &
                                    (ScrapedPage.id <= (last_id + batch_size)))\
                             .order_by(ScrapedPage.id.asc()))  # type: List[ScrapedPage]

        print('Fetched', len(scraped_pages))
        if len(scraped_pages) <= 0:
            return

        for page in scraped_pages:
            yield page

            last_id = page.id


def fetch_pages_jsonl():
    with open('data/7_opensources_co/db_fnr_2018_01_05.json', 'r', encoding='utf-8') as in_scraped:
        for i, line in enumerate(in_scraped):
            if i < 3976234:
                continue

            page = ujson.loads(line.strip()[1:-1])
            yield ScrapedPage(id=page['id'], batch=page['batch'], url=page['url'], html=page['html'],
                              inserted_at=page['inserted_at'], updated_at=page['updated_at'])


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    df_websites = pd.read_excel('data/7_opensources_co/websites_with_results.xlsx')
    domains = [u for u in df_websites.url.values]

    domain_type = {}
    websites_url = df_websites.url.values
    websites_type = df_websites.type.values
    for i, url in enumerate(websites_url):
        domain_type[url] = websites_type[i]

    urls_domains_not_found = []

    with open('data/7_opensources_co/news_cleaned_postgres_missing_domains.csv', 'w') as out_missing_domains:
        with tqdm() as progress:
            print('Cleaning')
            pages_parsed = []
            with multiprocessing.pool.Pool(processes=multiprocessing.cpu_count()) as pool:
                # for page in pool.imap_unordered(parse_article, fetch_pages(last_id, batch_size), chunksize=100):
                for page in pool.imap_unordered(parse_article, fetch_pages_jsonl(), chunksize=100):
                    if page is None:
                        continue

                    domain = None
                    for d in domains:
                        if d in page['url']:
                            domain = d

                    if domain is None:
                        urls_domains_not_found.append(page['url'])
                        out_missing_domains.write(page['url'] + '\n')

                        domain = urlsplit(page['url']).netloc

                    page['domain'] = domain
                    page['type'] = domain_type[page['domain']] if page['domain'] in domain_type else None

                    pages_parsed.append(page)
                    progress.update()

                    if len(pages_parsed) > 1000:
                        print('Inserting cleaned articles to DB')
                        with peewee_database.atomic():
                            Page.insert_many(pages_parsed).execute()
                            pages_parsed = []

            print('Inserting cleaned articles to DB')
            with peewee_database.atomic():
                Page.insert_many(pages_parsed).execute()

            print('Urls without our domains?!:', len(urls_domains_not_found))


if __name__ == '__main__':
    main()
