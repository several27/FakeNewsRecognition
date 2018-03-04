import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from machines.data_generator import path_data

"""
Some articles seem to have different urls but the same content, let's count the content hashes then.
"""


def main():
    csv.field_size_limit(500 * 1024 * 1024)

    path_data_articles = path_data + 'news_cleaned_2018_02_13.csv'

    counters = {
        'all': 0,
        'short_content': 0,
        'type_nan': 0
    }

    unique_hashes = {
        'title_url': set(),
        'title_content': set(),
        'url': set(),
        'content': set()
    }

    with tqdm() as progress:
        for df_articles in pd.read_csv(path_data_articles, engine='python', chunksize=10000):
            for article in df_articles.itertuples():
                unique_hashes['url'].add(article.url.__hash__())
                unique_hashes['content'].add(article.content.__hash__())

                unique_hashes['title_url'].add((article.title, article.url).__hash__())
                unique_hashes['title_content'].add((article.title, article.content).__hash__())

                if isinstance(article.type, str):
                    type_ = 'type_%s' % article.type
                    counters.setdefault(type_, 0)
                    counters[type_] += 1
                else:
                    counters['type_nan'] += 1

                if article.content is None or len(article.content) < 60:
                    counters['short_content'] += 1

                counters['all'] += 1
                progress.update()

    for k, v in counters.items():
        print('Counter %s: %s' % (k, v))

    for k, v in unique_hashes.items():
        print('Unique hashes count %s: %s' % (k, len(v)))


if __name__ == '__main__':
    main()
