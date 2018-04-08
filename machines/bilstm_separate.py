import ujson

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input, Dense, Bidirectional, CuDNNLSTM, merge
from keras.models import Model
from tqdm import tqdm

from machines.callbacks.save_to_spread import SaveToSpread
from machines.data_generator import path_data, path_news_shuffled, path_fasttext_jsonl, \
    embedded_news_generator_separate, path_news_train_all_separate, path_news_val_all_separate

embedding_size = 100
title_max_words = 50
input_shape_title = title_max_words, embedding_size
content_max_words = 250
input_shape_content = content_max_words, embedding_size

batch_size = 64
epochs = 10


def bilstm_separate_model(units_title=(64,), units_content=(64,), units=(64,), dropout=(0.5,), hidden_dims=18):
    batch_input_shape_title = (batch_size, input_shape_title[0], input_shape_title[1])
    batch_input_shape_content = (batch_size, input_shape_content[0], input_shape_content[1])

    model_input_title = Input(shape=input_shape_title, batch_shape=batch_input_shape_title)
    model_input_content = Input(shape=input_shape_content, batch_shape=batch_input_shape_content)

    bilstm_title = Bidirectional(CuDNNLSTM(units_title[0], return_sequences=True, stateful=True))(model_input_title)
    bilstm_content = Bidirectional(CuDNNLSTM(units_content[0], return_sequences=True, stateful=True))(
        model_input_content)

    merged = merge([bilstm_title, bilstm_content], mode='concat', concat_axis=-2)
    lstm = Bidirectional(CuDNNLSTM(units[0], stateful=True))(merged)
    previous_layer = Dropout(dropout[0])(lstm)

    z = Dense(hidden_dims, activation='relu')(previous_layer)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model([model_input_title, model_input_content], model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    with tf.device('/gpu:0'):
        model = bilstm_separate_model()
        model.summary()

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
        print('Loading fasttext...')
        fasttext_dict = {}
        with tqdm() as progress:
            with open(path_fasttext_jsonl, 'r') as in_fasttext:
                for line in in_fasttext:
                    embedding = ujson.loads(line)
                    fasttext_dict[embedding['word']] = np.asarray(embedding['embedding'])
                    progress.update()

        file_weights = 'bilstm_separate_weights.{epoch:03d}-{val_acc:.4f}.hdf5'
        params = 'units_title=(64,), units_content=(64,), units=(64,), dropout=(0.5,), hidden_dims=18'
        save_to_spread = SaveToSpread('2018_02_13', 'bilstm_separate.py', params, file_weights)

        train_generator = embedded_news_generator_separate(path_news_train_all_separate, batch_size, fasttext_dict,
                                                           title_max_words, content_max_words)
        test_generator = embedded_news_generator_separate(path_news_val_all_separate, batch_size, fasttext_dict,
                                                          title_max_words, content_max_words)

        checkpoint = ModelCheckpoint(path_data + file_weights, monitor='val_acc', verbose=1, mode='auto')
        model.fit_generator(train_generator, steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                            validation_data=test_generator, validation_steps=val_size // batch_size,
                            callbacks=[save_to_spread, checkpoint])


def test():
    print('Loading fasttext...')
    cnn_model = lstm_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
