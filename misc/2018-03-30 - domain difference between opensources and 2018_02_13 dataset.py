import pandas as pd


def main():
    path_data = 'data/7_opensources_co/'

    df_websites = pd.read_excel(path_data + 'websites_with_results.xlsx')
    df_websites_scraped = pd.read_csv(path_data + 'news_cleaned_2018_02_13_domains.csv')

    domains = set([d.lower().replace('www.', '') for d in df_websites['url']])
    domains_scraped = set([d.lower().replace('www.', '') for d in df_websites_scraped['domain']])

    domains_missing = domains - domains_scraped

    domain_type = dict([(d.url.lower().replace('www.', ''), (d.type, d.result))
                        for d in df_websites.itertuples()])
    df_websites_missing = pd.DataFrame([{'url': d, 'type': domain_type[d][0], 'result': domain_type[d][1]}
                                        for d in domains_missing])
    df_websites_missing.to_excel(path_data + 'websites_missing_2018_02_13.xlsx')


if __name__ == '__main__':
    main()
