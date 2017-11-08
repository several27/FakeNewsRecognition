import os
import re
import multiprocessing
from itertools import chain
from collections import deque
from typing import Tuple, Union

import ujson
import peewee
import requests
import newspaper
import tldextract
import pandas as pd
from tqdm import tqdm
from lxml.html import tostring
from lxml.html import fromstring
from lxml.cssselect import CSSSelector
from dotenv import load_dotenv, find_dotenv


class Parser:
    accepted_urls = '(.*)'
    selectors = ['title', 'content', 'published_at', 'authors']

    selector_title = None
    selector_content = None
    selector_published_at = None
    selector_authors = None

    def __init__(self, url, html=None):
        self.url = url
        self.html = html

        self.title = None
        self.content = None
        self.published_at = None
        self.authors = None

        self.all_urls = None
        self.children_urls = None

        self.accepted_urls_compiled = re.compile(self.accepted_urls)

    def parse(self):
        use_selectors = False
        use_newspaper = False
        for selector, selector_value in self.get_selectors_values().items():
            if selector_value is None:
                use_newspaper = True
            else:
                use_selectors = True

        if use_newspaper:
            article = newspaper.Article(self.url)
            article.set_html(self.html)
            article.parse()
            self.title = article.title
            self.content = article.text
            self.published_at = article.publish_date
            self.authors = article.authors

        html_tree = fromstring(self.html)
        if use_selectors:
            for selector, selector_value in self.get_selectors_values().items():
                if callable(selector_value):
                    setattr(self, selector, selector_value(html_tree))
                else:
                    css_selector = CSSSelector(selector_value)
                    result = css_selector(html_tree)
                    if len(result) == 1:
                        value = self._stringify_children(result[0])
                    elif len(result) > 1:
                        value = [self._stringify_children(r) for r in result]
                    else:
                        value = [self._stringify_children(r) for r in result]

                    setattr(self, selector, value)

        urls = CSSSelector('a')(html_tree)
        self.all_urls = [u.get('href') for u in urls if u.get('href') is not None]
        self.children_urls = filter(self.filter_children_url, self.all_urls)
        self.children_urls = set(map(self.map_children_url, self.children_urls))

    def filter_children_url(self, url: str) -> bool:
        return (url.startswith('/') and not url.startswith('//')) or \
               tldextract.extract(url) == tldextract.extract(self.url)

    def map_children_url(self, url: str) -> str:
        return url.split('#', 1)[0]

    def _stringify_children(self, node):
        parts = ([node.text] +
                 list(chain(*([c.text, tostring(c), c.tail] for c in node.getchildren()))) +
                 [node.tail])

        return ''.join(filter(None, parts))

    def get_selectors_values(self):
        selectors_values = {}
        for selector in self.selectors:
            selectors_values[selector] = getattr(self, 'selector_' + selector)

        return selectors_values

    def accept_url(self):
        return self.accepted_urls_compiled.match(self.url)


database_proxy = peewee.Proxy()


class BaseModel(peewee.Model):
    class Meta:
        database = database_proxy


class Schedule(BaseModel):
    url = peewee.CharField(null=False, max_length=4000)
    finished = peewee.BooleanField(null=False, default=False)


class Results(BaseModel):
    url = peewee.CharField(null=False, max_length=4000)
    status = peewee.CharField(null=False)
    title = peewee.CharField(null=True)
    published_at = peewee.CharField(null=True)
    content = peewee.TextField(null=True)
    authors = peewee.TextField(null=True)
    html = peewee.TextField(null=True)
    children_urls = peewee.TextField(null=True)


class Scraper:
    def __init__(self, websites, parsers=None, processes=None):
        self.websites = websites
        self.parsers = (parsers + [Parser]) if parsers is not None else [Parser]
        self.processes = processes if processes is not None else multiprocessing.cpu_count() * 4

        Schedule.create_table(fail_silently=True)
        Results.create_table(fail_silently=True)

        if Schedule.select().count() == 0:
            for url in websites:
                Schedule.insert(url=url).execute()

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/61.0.3163.100 Safari/537.36'
        }

    def run(self):
        # print('Scheduled', len(self.schedule), 'websites')

        handled_urls = set()
        bar = tqdm(total=self.count_schedule())
        pool = multiprocessing.Pool(processes=self.processes)

        for _id, website, result in pool.imap(self.handle_request, self.yield_schedule(), chunksize=1):
            handled_urls.add(website)

            s = Schedule.get(Schedule.id == _id)
            s.finished = True
            s.save()

            if result is None:
                Results.insert(url=website, status='failed').execute()
                continue

            Results.insert(url=website, status='success', title=result.title, published_at=result.published_at,
                           content=result.content, authors=result.authors, html=result.html,
                           children_urls=result.children_urls).execute()

            children_urls = [u for u in result.children_urls if u not in handled_urls]

            handled_urls.add(result.url)
            handled_urls.update(set(children_urls))

            for url in children_urls:
                Schedule.insert(url=url).execute()

            print('Scheduled additional', len(children_urls), 'websites')

            bar.total += len(children_urls)
            bar.update()

    def yield_schedule(self):
        while True:
            schedule = list(Schedule.select().where(~Schedule.finished).order_by(Schedule.id.desc()).limit(5))
            if len(schedule) <= 0:
                return

            for s in schedule:
                yield (s.id, self.get_parser(s.url).__class__, s.url)

    def get_parser(self, url):
        for parser in self.parsers:
            p = parser(url)
            if p.accept_url():
                return p

        return None

    def count_schedule(self):
        return Schedule.select(Schedule.finished).where(~Schedule.finished).count()

    def handle_request(self, request) -> Tuple[int, str, Union[None, Parser]]:
        _id, parser, website = request
        parser = parser(website)

        print('Parsing website', website)

        result = requests.get(website, headers=self.headers)
        if result.status_code != 200:
            return _id, website, None

        parser.url = result.url
        parser.html = result.content
        parser.parse()

        return _id, website, parser


def main():
    load_dotenv(find_dotenv())
    path_fnr = os.environ['PATH_FNR']
    path_7_fn = path_fnr + 'data/7_fake_news/'

    path_websites = path_7_fn + 'websites_with_results.xlsx'

    df_websites = pd.read_excel(path_websites)
    websites_url = list(reversed(['http://' + u for u in df_websites[df_websites.result == 200].url.values]))

    print('Found', len(websites_url), '/', len(df_websites), 'websites to scrape!')

    path_db = path_7_fn + 'scraper.db'
    database = peewee.SqliteDatabase(path_db)
    database_proxy.initialize(database)

    scraper = Scraper(websites_url, processes=1)
    scraper.run()


if __name__ == '__main__':
    main()
