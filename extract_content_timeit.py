import timeit
from typing import List

from tqdm import tqdm
import newspaper

from database import ScrapedPage


def main():
    scraped_page = (ScrapedPage.select().where(ScrapedPage.batch == 2)
        .order_by(+ScrapedPage.id.asc())
        .limit(1))[0]  # type: ScrapedPage

    def newspaper_parse():
        article = newspaper.Article(scraped_page.url)
        article.set_html(scraped_page.html)
        article.parse()

    print('Starting timing url:', scraped_page.url)
    print(min(timeit.Timer(stmt=newspaper_parse).repeat(3, 100)))

    """
    Results: 
    Starting timing url: %s http://awarenessact.com/bomb-cyclone-hits-east-coast-florida-reaches-temperatures-lower-than-alaska/
    14.771097040968016
    """


if __name__ == '__main__':
    main()
