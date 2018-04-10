import tensorflow as tf
import ujson
import numpy as np
from gensim.models.fasttext import FastText
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input, Dense, Bidirectional, CuDNNLSTM, Embedding
from keras.models import Model
from tqdm import tqdm

from machines.data_generator import embedded_news_generator, path_data, path_news_train, path_news_val, path_fasttext, \
    path_news_shuffled, path_fasttext_jsonl, non_embedded_news_generator

max_words = 300
input_shape = max_words,

batch_size = 64
epochs = 10


def bilstm_model(index_dict, word_vectors, units=(64,), dropout=(0.5,), hidden_dims=17):
    batch_input_shape = (batch_size, input_shape[0])
    model_input = Input(shape=input_shape, batch_shape=batch_input_shape)

    n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols, 100))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    embedding_layer = Embedding(output_dim=100, input_dim=n_symbols, trainable=True)
    embedding_layer.build((None,))  # if you don't do this, the next step won't work
    embedding_layer.set_weights([embedding_weights])
    embedded = embedding_layer(model_input)

    previous_layer = embedded
    for i, u in enumerate(units):
        if i != (len(units) - 1):
            lstm = Bidirectional(CuDNNLSTM(u, return_sequences=True, stateful=True))(previous_layer)
        else:
            lstm = Bidirectional(CuDNNLSTM(u, stateful=True))(previous_layer)

        previous_layer = Dropout(dropout[i])(lstm)

    z = Dense(hidden_dims, activation='relu')(previous_layer)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    print('Loading fasttext...')
    fasttext_dict = {}
    index_dict = {}
    with tqdm() as progress:
        with open(path_fasttext_jsonl, 'r') as in_fasttext:
            for index, line in enumerate(in_fasttext):
                embedding = ujson.loads(line)
                fasttext_dict[embedding['word']] = np.asarray(embedding['embedding'])
                index_dict[embedding['word']] = index
                progress.update()

    with tf.device('/gpu:0'):
        cnn_model = bilstm_model(index_dict, fasttext_dict, units=(128, 64, 32), dropout=(.2, .2, .2), hidden_dims=18)
        cnn_model.summary()

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
        checkpoint = ModelCheckpoint(path_data + 'bilstm_stacked_512_weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                     monitor='val_acc', verbose=1, mode='auto')
        cnn_model.fit_generator(non_embedded_news_generator(path_news_train, batch_size, index_dict, max_words),
                                steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                                validation_data=non_embedded_news_generator(path_news_val, batch_size, index_dict,
                                                                            max_words),
                                validation_steps=val_size // batch_size, callbacks=[checkpoint])


def test():
    print('Loading fasttext...')
    cnn_model = lstm_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
