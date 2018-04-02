from typing import Union, List

import gensim
import numpy as np
import time
from keras.engine import Input
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Conv1D, Activation, Flatten, Add
from keras.models import Model
from peewee import DoesNotExist

from api.database import WordEmbedding

max_words = 300
input_shape = max_words, 100
news_labels = ['bias', 'clickbait', 'conspiracy', 'fake', 'hate', 'junksci', 'political', 'reliable', 'rumor',
               'satire', 'unreliable']


def bilstm_model(units=64, dropout=(0.5,), hidden_dims=17):
    model_input = Input(shape=input_shape)

    previous_layer = model_input
    for u in ([units] if not isinstance(units, list) else units):
        previous_layer = Bidirectional(LSTM(u))(model_input)

    z = Dropout(dropout[0])(previous_layer)
    z = Dense(hidden_dims, activation='relu')(z)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def bilstm_all_model(units: Union[List[int], int] = 64, dropout=(0.5,), hidden_dims=17):
    model_input = Input(shape=input_shape)

    previous_layer = model_input
    for u in ([units] if not isinstance(units, list) else units):
        previous_layer = Bidirectional(LSTM(u))(model_input)

    z = Dropout(dropout[0])(previous_layer)
    z = Dense(hidden_dims, activation='relu')(z)
    model_output = Dense(len(news_labels), activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def TemporalBlock(input_, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    conv_0 = Conv1D(n_outputs, kernel_size=1)(input_)

    conv_1 = Conv1D(n_outputs, kernel_size, strides=stride, padding='same', dilation_rate=dilation)(input_)
    # chomp_2 = Chomp1D(padding)(conv_1)
    relu_3 = Activation('relu')(conv_1)
    dropout_4 = Dropout(dropout)(relu_3)

    conv_5 = Conv1D(n_outputs, kernel_size, strides=stride, padding='same', dilation_rate=dilation)(dropout_4)
    # chomp_6 = Chomp1D(padding)(conv_5)
    relu_7 = Activation('relu')(conv_5)
    dropout_8 = Dropout(dropout)(relu_7)

    add_9 = Add()([dropout_8, conv_0])
    relu_10 = Activation('relu')(add_9)

    return relu_10


def tcn_all_model(num_channels, kernel_size=2, dropout=0.2):
    model_input = Input(shape=input_shape)

    previous_layer = model_input
    for i in range(len(num_channels)):
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        previous_layer = TemporalBlock(previous_layer, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size - 1) * dilation_size, dropout=dropout)

    flatten = Flatten()(previous_layer)
    model_output = Dense(len(news_labels), activation='softmax')(flatten)

    model = Model(model_input, model_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def embed(content: str, fasttext_dict = None) -> np.ndarray:
    content_cleaned = gensim.parsing.preprocessing.preprocess_string(content, [
        gensim.parsing.preprocessing.strip_tags,
        gensim.parsing.preprocessing.strip_punctuation,
        gensim.parsing.preprocessing.strip_multiple_whitespaces,
        gensim.parsing.preprocessing.strip_numeric,
        gensim.parsing.preprocessing.remove_stopwords,
        gensim.parsing.preprocessing.strip_short,
        lambda x: x.lower()
    ])

    i = 0
    embedding = np.zeros(input_shape)
    for word in content_cleaned:
        if fasttext_dict is None:
            try:
                word_embedding = WordEmbedding.get(
                    (WordEmbedding.version == 1) & (WordEmbedding.word == word))  # type: WordEmbedding
            except DoesNotExist:
                continue

            embedding[i] = word_embedding.embedding
        else:
            if word in fasttext_dict:
                embedding[i] = fasttext_dict[word]
            else:
                continue

        i += 1

        if i >= max_words:
            break

    return embedding
