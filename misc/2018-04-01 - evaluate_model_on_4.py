import numpy as np
import pandas as pd
import ujson
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from api.helpers import bilstm_model, embed, input_shape
from machines.data_generator import path_fasttext_jsonl


def main():
    df_news = pd.read_csv('data/4_fake_real_news_dataset/fake_or_real_news.csv')
    print(df_news.shape)

    model = bilstm_model()
    model.load_weights('api/bilstm_weights.010-0.9848.hdf5')

    y_test = []

    print('Loading fasttext...')
    fasttext_dict = {}
    with tqdm() as progress:
        with open(path_fasttext_jsonl, 'r') as in_fasttext:
            for line in in_fasttext:
                embedding = ujson.loads(line)
                fasttext_dict[embedding['word']] = np.asarray(embedding['embedding'])
                progress.update()

    embeddings = []
    with tqdm() as progress:
        for article in list(df_news.itertuples()):
            embedding = embed(article.title + article.text, fasttext_dict)
            embeddings.append(embedding)
            y_test.append(article.label)
            progress.update()

    predictions = model.predict(np.asarray(embeddings))
    y_pred = ['FAKE' if pred <= 0.5 else 'REAL' for pred in predictions]

    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # [[2687  477] [1422 1749]]
    # 0.70
    # Not too bad, but not good either :P


if __name__ == '__main__':
    main()
