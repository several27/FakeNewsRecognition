# Project Brief
Student’s Name: Maciej Szpakowski (ms3u14 - 28170911)
Project Title: Fake News Recognition
Supervisor’s Name: Dr. Jonathon Hare 

## Problem

Nowadays, with the increased ease of online communication and social media, people have immediate access to many different types of information. Some of this information comes from reliable portals. However, there's plenty of news coming from gossipy magazines and click baits that share scientifically dubious claims, hatred and biased speech. The problem becomes even more prominent as people often share articles that have controversial titles, increasing their popularity. Unfortunately, with more sources appearing on the internet, manual validation is becoming impossible. 

## Goals

The goal of this project is to explore the issue of fake news and create a deep learning based algorithm for automatic labelling of texts that are potentially deceptive. As this is a far-reaching problem that affects many areas and different types of media, the first step is to narrow it down based on available data and previous research. 

The second step is to, using this data, create/choose the most efficient algorithm that will be able to determine the veracity of the news. Some of the possible approaches include context-based (e.g., using the links network), knowledge-based (e.g., using information from Wikipedia), style-based (e.g., using text classification) or an amalgamation of those. 

Additionally, if time allows, the last step is to create a simple browser extension, which will automatically detect fake news and inform the user about it (e.g., by showing an alert on the feed or actual news page). It will also allow for gathering feedback from users to assess the real performance of created approach.

## Scope 

One of the most significant limitations of this work is the data that is openly available. The focus of this work is on using novel deep learning based approaches for natural language processing. Some of them (like LSTMs based on word embeddings) usually require large quantities of data. Therefore additional website crawling may be necessary. 

Finally, the term “fake news” sometimes is associated with political views some people do not agree with. While we will do our best to make sure the datasets used for training contain actual information and are accurately classified, the purpose of this work is not to fact check all the input data to make sure initial labels are 100% correct.

# Data pipeline

1. Run the scraper. From inside of the spider directory run:  

    ```bash
    export PATH_FNR=/home/several27/FakeNewsRecognition/
    scrapy crawl news -s LOG_FILE="../data/7_opensources_co/news_spider.log.1" -s JOBDIR="../data/7_opensources_co/news_spider_job_1/"
    ```

2. Parse the scraped dataset by creating another DB with title and content limiters 
2. Download the webhose dataset and convert to a DB (`Webhose analytics.ipynb`) 
3. Download the nytimes dataset (`scrape_nytimes.py`)
4. Append the webhose and nytimes datasets to the scraped one setting the type to reliable and source to webhose and nytimes (`news_cleaned + webhose + nytimes merge.ipynb`)
5. Convert the dataset to csv (`data_full.py`)
6. Create a dataset sample (`data_sample.py`)
7. Generate fasttext embeddings (`Generate fastText.ipynb`)

# Running machines 

To run the machines the supplied csv from FakeNewsCorpus needs to preprocessed using the data_generator.py. The fasttext embeddings are also necessary.

# Look into those counts

ms3u14@myrtle:~/FakeNewsRecognition$ python3 2018-02-21\ -\ count_unique_articles.py 
8529090it [14:14, 9987.15it/s] 
Counter all: 8529090
Counter short_content: 0
Unique hashes count content: 6248655
Unique hashes count url: 8462558
Unique hashes count title_url: 8465409
Unique hashes count title_content: 7297002