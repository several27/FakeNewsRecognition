import tensorflow as tf
import ujson
import numpy as np
from gensim.models.fasttext import FastText
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input, Dense, Bidirectional, CuDNNLSTM, Flatten, Activation, RepeatVector, Permute, \
    merge, K, Lambda
from keras.models import Model
from tqdm import tqdm

from machines.callbacks.save_to_spread import SaveToSpread
from machines.data_generator import embedded_news_generator, path_data, path_news_train, path_news_val, path_fasttext, \
    path_news_shuffled, path_fasttext_jsonl
from machines.layers.attention_lstm import attention_3d_block

max_words = 300
input_shape = max_words, 100

batch_size = 1024
epochs = 10


def bilstm_attention_model(units=(64,), dropout=(0.5,), hidden_dims=17):
    batch_input_shape = (batch_size, input_shape[0], input_shape[1])
    model_input = Input(shape=input_shape, batch_shape=batch_input_shape)

    previous_layer = model_input
    for i, u in enumerate(units):
        if i != (len(units) - 1):
            previous_layer = Bidirectional(CuDNNLSTM(u, return_sequences=True, stateful=True))(previous_layer)
        else:
            previous_layer = Bidirectional(CuDNNLSTM(u, return_sequences=True, stateful=True))(previous_layer)

    attention = Dense(1, activation='tanh')(previous_layer)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units[-1] * 2)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = merge([previous_layer, attention], mode='mul')
    sent_representation = Lambda(lambda x: K.mean(x, axis=-2), output_shape=(units[-1] * 2,))(sent_representation)

    z = Dense(hidden_dims, activation='relu')(sent_representation)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    with tf.device('/gpu:0'):
        cnn_model = bilstm_attention_model(units=(128, 64, 32,), dropout=(.2, .2, .2, .2, .1, .1, .1),
                                           hidden_dims=18)
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
        print('Loading fasttext...')
        fasttext_dict = {}
        with tqdm() as progress:
            with open(path_fasttext_jsonl, 'r') as in_fasttext:
                for line in in_fasttext:
                    embedding = ujson.loads(line)
                    fasttext_dict[embedding['word']] = np.asarray(embedding['embedding'])
                    progress.update()

        checkpoint = ModelCheckpoint(path_data + 'bilstm_attention_s_weights.{epoch:03d}-{val_acc:.4f}.hdf5',
                                     monitor='val_acc', verbose=1, mode='auto')
        save_to_spread = SaveToSpread('2018_02_13', 'bilstm_attention.py', 'units=(32,), dropout=(.2), hidden_dims=18',
                                      'bilstm_attention_s_weights.{epoch:03d}-{val_acc:.4f}.hdf5')
        cnn_model.fit_generator(embedded_news_generator(path_news_train, batch_size, fasttext_dict, max_words),
                                steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                                validation_data=embedded_news_generator(path_news_val, batch_size, fasttext_dict,
                                                                        max_words),
                                validation_steps=val_size // batch_size, callbacks=[checkpoint, save_to_spread])


def test():
    print('Loading fasttext...')
    cnn_model = lstm_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
