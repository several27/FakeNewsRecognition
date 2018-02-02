import pandas as pd

from database import peewee_database


def main():
    count = 1 * 1000 * 1000
    sample = 250

    probability, limit = count / sample, sample

    connection = peewee_database.get_conn()
    cursor = connection.cursor()
    cursor.execute('select * from fnr_pages where '
                   'length(url) > 50 and length(content) > 50 and random() %% %s == 0 limit %s' % (probability, limit))
    results = cursor.fetchall()

    pages_sample = []
    for result in results:
        row = {}
        for i, column in enumerate(result):
            row[cursor.description[i][0]] = column

        pages_sample.append(row)

    df_pages = pd.DataFrame([p for p in pages_sample], columns=['id', 'domain', 'type', 'url', 'content', 'scraped_at',
                                                                'inserted_at', 'updated_at', 'title', 'authors',
                                                                'keywords', 'meta_keywords', 'meta_description',
                                                                'tags', 'summary'])
    df_pages.to_csv('data/7_opensources_co/news_cleaned_2018_01_29_sample.csv')


if __name__ == '__main__':
    main()
