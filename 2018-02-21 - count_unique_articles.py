import csv
import pandas as pd
from tqdm import tqdm

"""
Some articles seem to have different urls but the same content, let's count the content hashes then.
"""


def main():
    csv.field_size_limit(500 * 1024 * 1024)

    data_path = '/Volumes/ExternalSSD/FakeNewsRecognition/'
    data_articles = data_path + 'news_cleaned_2018_02_13.csv'

    articles_count = 0
    unique_articles = set()
    with tqdm() as progress:
        for df_articles in pd.read_csv(data_articles, engine='python', chunksize=10000):
            for article in df_articles.itertuples():
                unique_articles.add(article.content.__hash__())
                articles_count += 1
                progress.update()

    print('Unique hashes:', len(unique_articles), 'out of', articles_count)


if __name__ == '__main__':
    main()
