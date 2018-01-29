from typing import List

from peewee import fn
from tqdm import tqdm
import newspaper

from database import Page, ScrapedPage


def main():
    # scraped_count = ScrapedPage.select(fn.Count(ScrapedPage.id)).where(ScrapedPage.batch == 2)

    with tqdm() as progress:
        last_id = 0
        while True:
            scraped_pages = ScrapedPage.select().where((ScrapedPage.batch == 2) & (ScrapedPage.id > last_id))\
                .order_by(+ScrapedPage.id.asc())\
                .limit(10)  # type: List[ScrapedPage]

            for page in scraped_pages:
                article = newspaper.Article(page.url)
                article.set_html(page.html)
                article.parse()

                last_id = page.id
                progress.update()


if __name__ == '__main__':
    main()
