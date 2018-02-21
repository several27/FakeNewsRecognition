import csv

import tensorflow as tf
from gensim.models.fasttext import FastText
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate, Dropout, Activation, Flatten, Input, Dense, Conv1D, MaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from machines.data_generator import embedded_news_generator, path_data, path_news_train, path_news_val, path_fasttext, \
    path_news_shuffled

"""
From "2018-02-17 - FakeNewsCorpus Simple CNN.ipynb" notebook
"""

csv.field_size_limit(500 * 1024 * 1024)

path_news_embedded = path_data + 'news_cleaned_2018_02_13.embedded.jsonl'

max_words = 300
input_shape = max_words, 100

batch_size = 64
epochs = 5


def cnn_deep_model(filters=512, drop=0.5, filter_sizes=(3, 4, 5)):
    # https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py

    inputs = Input(shape=input_shape)

    maxpools = []
    for filter_size in filter_sizes:
        conv = Conv1D(filters, kernel_size=filter_size, padding='valid',
                      kernel_initializer='normal', activation='relu')(inputs)
        maxpool = MaxPool1D(pool_size=max_words - filter_size + 1, strides=1,
                            padding='valid')(conv)
        maxpools.append(maxpool)

    concatenated_tensor = Concatenate(axis=1)(maxpools)
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    activation = Activation('sigmoid')(dropout)
    output = Dense(1)(activation)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def cnn_deep_model_2(filters=10, dropout_prob=(0.5, 0.8), filter_sizes=(3, 8), hidden_dims=50):
    # https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py
    # https://arxiv.org/pdf/1408.5882v2.pdf

    model_input = Input(shape=input_shape)
    z = Dropout(dropout_prob[0])(model_input)

    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(filters=filters, kernel_size=sz, padding='valid', activation='relu', strides=1)(z)
        conv = MaxPool1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation='relu')(z)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    print('Loading fasttext...')
    fasttext = FastText.load_fasttext_format(path_fasttext)

    print('Counting input...')
    count_lines = 0
    with open(path_news_shuffled, 'r') as in_news:
        for _ in tqdm(in_news):
            count_lines += 1

    train_size = int(count_lines * .8)
    test_size = int(count_lines * .8)
    val_size = count_lines - (int(count_lines * 0.8) + int(count_lines * 0.1))

    print('Train size:', train_size, '; test size:', test_size, '; val size:', val_size)

    print('Training...')
    with tf.device('/gpu:0'):
        cnn_model = cnn_deep_model_2()
        checkpoint = ModelCheckpoint(path_data + 'cnn_deep_2_weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                                     verbose=1, mode='auto')
        cnn_model.fit_generator(embedded_news_generator(path_news_train, batch_size, fasttext, max_words),
                                steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                                validation_data=embedded_news_generator(path_news_val, batch_size, fasttext, max_words),
                                validation_steps=val_size // batch_size, callbacks=[checkpoint])


def test():
    print('Loading fasttext...')
    cnn_model = cnn_deep_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
