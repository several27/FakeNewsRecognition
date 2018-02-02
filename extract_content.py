from typing import List

import multiprocessing
import multiprocessing.pool
from tqdm import tqdm
import newspaper

from database import Page, ScrapedPage, peewee_database


def parse_article(scraped_page: ScrapedPage):
    if len(scraped_page.html) <= 0:
        print('Missing url for id', scraped_page.id)
        return None

    try:
        article = newspaper.Article(scraped_page.url, fetch_images=False)
        article.set_html(scraped_page.html)
        article.parse()
    except newspaper.article.ArticleException:
        print('Missing url for id', scraped_page.id)
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
        scraped_pages = list(ScrapedPage.select() \
                             .where((ScrapedPage.batch == 2) & (ScrapedPage.id > last_id) &
                                    (ScrapedPage.id <= (last_id + batch_size))) \
                             .order_by(ScrapedPage.id.asc()))  # type: List[ScrapedPage]

        print('Fetched', len(scraped_pages))
        if len(scraped_pages) <= 0:
            return

        for page in scraped_pages:
            yield page

            last_id = page.id


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    with tqdm() as progress:
        last_id = 6756015  # last batch = 1 in the sqlite DB
        batch_size = 10 * 1000

        print('Cleaning')
        pages_parsed = []
        with multiprocessing.pool.Pool(processes=multiprocessing.cpu_count()) as pool:
            for page in pool.imap_unordered(parse_article, fetch_pages(last_id, batch_size), chunksize=100):
                if page is None:
                    continue

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


if __name__ == '__main__':
    main()
