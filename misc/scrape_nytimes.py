# https://api.nytimes.com/svc/archive/v1/2016/1.json?api-key=29a51170349f43d9abe651b0e2331ea6
from time import sleep

import requests
import ujson

from tqdm import tqdm


def download_urls_from_archive():
    url = 'https://api.nytimes.com/svc/archive/v1/%s/%s.json?api-key=29a51170349f43d9abe651b0e2331ea6'

    year_from = 2000  # 1852
    year_to = 2018

    with open('data/nytimes/archive.json', 'w') as out_archive:
        with tqdm(total=(year_from - year_to) * 12) as progress:
            for year in range(year_from, year_to + 1):
                for month in range(1, 13):
                    docs = None
                    try:
                        response = requests.get(url % (year, month))
                        docs = response.json()['response']['docs']
                    except Exception:
                        sleep(10)

                    if docs is None:
                        response = requests.get(url % (year, month))
                        docs = response.json()['response']['docs']

                    for doc in docs:
                        out_archive.write(ujson.dumps(doc) + '\n')

                    progress.update()


def main():
    pass


if __name__ == '__main__':
    main()
