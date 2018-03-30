import os
import multiprocessing

import ujson
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

from cleaner import Cleaner
from tagger import Tagger


class FNCDataPreProcessor:
    word2vec_model = None

    def __init__(self, path_bodies, path_stances, path_word2vec='/home/ubuntu/GoogleNews-vectors-negative300.bin'):
        self.path_bodies = path_bodies
        self.path_stances = path_stances
        self.path_word2vec = path_word2vec

        self.bodies_raw = None
        self.stances_raw = None

        self.path_bodies_tagged = self._switch_path_ext(self.path_bodies, '_tagged.json')
        self.path_stances_tagged = self._switch_path_ext(self.path_stances, '_tagged.json')

        self.bodies_tagged = None
        self.stances_tagged = None

        self.word_vec_model = None

        self.bodies_vec = None
        self.stances_vec = None
        self.bodies_vec_missed_words = None
        self.stances_vec_missed_words = None

    def load_raw(self):
        print('Loading raw data...')

        self.bodies_raw = pd.read_csv(self.path_bodies)
        self.stances_raw = pd.read_csv(self.path_stances)

    def load_tagged(self):
        print('Loading tagged data...')

        self.bodies_tagged = self._cached(self.path_bodies_tagged, self.clean_and_tag,
                                          self.bodies_raw['articleBody'])
        self.stances_tagged = self._cached(self.path_stances_tagged, self.clean_and_tag,
                                           self.stances_raw['Headline'])

    def clean_and_tag(self, data):
        print('Cleaning and tagging', len(data), 'records...')

        return Tagger.batch_perform(Cleaner.batch_perform(data), multiprocessing.cpu_count(), 100)

    def load_vectors(self):
        print('Creating word embeddings...')
        self.load_word_vec_model()

        self.bodies_vec, self.bodies_vec_missed_words = self.tagged_to_vectors(self.bodies_tagged)
        self.stances_vec, self.stances_vec_missed_words = self.tagged_to_vectors(self.stances_tagged)

    def load_word_vec_model(self):
        if FNCDataPreProcessor.word2vec_model is not None:
            print('Word2vec model loaded')
            self.word_vec_model = FNCDataPreProcessor.word2vec_model
            return self.word_vec_model

        print('Loading word2vec model')
        self.word_vec_model = KeyedVectors.load_word2vec_format(self.path_word2vec, binary=True)
        FNCDataPreProcessor.word2vec_model = self.word_vec_model

    def tagged_to_vectors(self, tagged):
        vec = []
        missed = 0
        for words in tagged:
            words_vec = [self.word_vec_model[w] for w in words if w in self.word_vec_model]
            vec.append(words_vec)

            missed += len(words) - len(words_vec)

        return vec, missed

    def training_data(self, feature_vec_body_size=500, feature_vec_headline_size=20, perform_train_test_split=0.9):
        self.load_raw()
        self.load_tagged()
        self.load_vectors()

        print('Constructing training vectors...')
        x_train = np.zeros((len(self.stances_vec), feature_vec_body_size + feature_vec_headline_size + 1, 300),
                           dtype=np.float32)
        y_train = np.zeros((len(self.stances_vec), len(set(self.stances_raw['Stance']))))

        for stance_id, headline, body_id, stance in tqdm(self.stances_raw.itertuples()):
            headline = np.array(self.stances_vec[stance_id])
            if headline.shape[0] == 0:
                headline = np.zeros((0, 300))

            x_train_row = np.concatenate((headline, np.zeros((0, 300)), (self.bodies_vec[self.bodies_raw.loc[
                self.bodies_raw['Body ID'] == body_id].index.values[0]])[:feature_vec_body_size - 1]), axis=0)
            x_train[stance_id][:x_train_row.shape[0]] = x_train_row

        possible_stances = list(set(self.stances_raw['Stance']))
        for idx, stance in enumerate(self.stances_raw['Stance']):
            y_train[idx][possible_stances.index(stance)] = 1

        if perform_train_test_split:
            return train_test_split(x_train, y_train, train_size=perform_train_test_split, shuffle=False)

        return x_train, y_train

    def _cached(self, path, fn, data):
        if not os.path.exists(path):
            with open(path, 'w') as _out:
                result = fn(data)
                ujson.dump(result, _out)
                return result

        with open(path, 'r') as _in:
            return ujson.load(_in)

    def _switch_path_ext(self, path, new_ext):
        return '%s%s' % (''.join(list(os.path.splitext(path)[:-1])), new_ext)
