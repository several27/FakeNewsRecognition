import ujson
from array import array
from typing import NamedTuple

import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from peewee import SqliteDatabase, Model, CharField, BlobField

from machines.bilstm import bilstm_model
from machines.cnn_deep import cnn_deep_model_2

path_data = 'data/fake_news_corpus/'
path_qualitative_test = path_data + 'qualitative_test.json'
path_model_weights_bilstm = path_data + 'bilstm_weights.010-0.9848.hdf5'
path_model_weights_cnn_deep_2 = path_data + 'cnn_deep_2_filter_sizes_2_3_4_hidden_5_filters_128_dropout_prob_08_0_weights.010-0.9288.hdf5'
path_db_embeddings = path_data + 'fasttext.db'

db_embeddings = SqliteDatabase(path_db_embeddings)


class BaseEmbeddingModel(Model):
    class Meta:
        database = db_embeddings


class Embedding(BaseEmbeddingModel):
    word = CharField()
    emb = BlobField()

    @property
    def embedding(self):
        return np.asarray(array('f', self.emb).tolist())

    @staticmethod
    def embed(content):
        count = 0
        embeddings = np.zeros((300, 100))
        for word in preprocess_string(content):
            embedding = Embedding.select().where(Embedding.word == word).execute()
            if len(embedding) > 0:
                embeddings[count] = embedding[0].embedding
                count += 1

        return np.asarray(embeddings)

    class Meta:
        db_table = 'embeddings'
        primary_key = False


ArticleTest = NamedTuple('ArticleTest', [('url', str), ('label', str), ('content', str)])


def main():
    with open(path_qualitative_test, 'r', encoding='utf-8') as in_test:
        test_articles = [ArticleTest(**a) for a in ujson.load(in_test)]

    machines = [
        (cnn_deep_model_2, dict(filter_sizes=(2, 3, 4), hidden_dims=5, filters=128, dropout_prob=(0.8, 0)),
         path_model_weights_cnn_deep_2),
        (bilstm_model, {}, path_model_weights_bilstm),
    ]

    for machine_fn, machine_params, machine_weights in machines:
        machine = machine_fn(**machine_params)
        machine.load_weights(machine_weights)

        print('Results for machine: %s with params: %s' % (machine_fn.__name__, machine_params))

        results = []
        for article in test_articles:
            article_emb = Embedding.embed(article.content)
            article_emb = article_emb.reshape(1, article_emb.shape[0], article_emb.shape[1])

            label_predicted_score = machine.predict(article_emb)[0][0]
            label_predicted = 'fake' if label_predicted_score < 0.5 else 'reliable'
            label_true = article.label

            print('%s\t%s\t%s' % (label_predicted_score, label_predicted, label_true))

            results.append(label_predicted == label_true)

        print('%s/%s' % (sum(results), len(results)))


if __name__ == '__main__':
    main()
