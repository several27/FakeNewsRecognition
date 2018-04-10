import pandas as pd
import ujson
from pandas import DataFrame

from machines.data_generator import path_news_csv, path_news_cleaned


def main():
    with open(path_news_cleaned + '_random_articles_labelled.jsonl', 'w') as _out:
        for df_news_chunk in pd.read_csv(path_news_csv, chunksize=10000):
            df_news_sampled = df_news_chunk.sample(frac=0.001)  # type: DataFrame
            for article in df_news_sampled.itertuples():
                print('\n\n---------------------------------------')
                print(article.url)
                print(article.title)
                print(article.content[:200])

                label = input('Good?:')
                _out.write(ujson.dumps({'url': article.url, 'label': label}) + '\n')


if __name__ == '__main__':
    main()
