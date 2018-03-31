import ujson

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.engine import Layer
from keras.layers import Dropout, Input, Conv1D, Activation, Add, Dense, Flatten
from keras.models import Model
from tqdm import tqdm

from machines.data_generator import embedded_news_generator, path_data, path_news_train, path_news_val, \
    path_news_shuffled, path_fasttext_jsonl

max_words = 300
input_shape = max_words, 100

batch_size = 64
epochs = 20


class Chomp1D(Layer):
    def __init__(self, chmop_size, **kwargs):
        self.chmop_size = chmop_size
        super().__init__(**kwargs)

    def build(self, input_shape_):
        super().build(input_shape_)

    def call(self, x, mask=None):
        return x[:, :, :-self.chmop_size]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2] - self.chmop_size


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


def tcn_model(num_channels, kernel_size=2, dropout=0.2):
    model_input = Input(shape=input_shape)

    previous_layer = model_input
    for i in range(len(num_channels)):
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        previous_layer = TemporalBlock(previous_layer, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size - 1) * dilation_size, dropout=dropout)

    flatten = Flatten()(previous_layer)
    model_output = Dense(1, activation='sigmoid')(flatten)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    print('Counting input...')
    count_lines = 0
    with open(path_news_shuffled, 'r') as in_news:
        for _ in tqdm(in_news):
            count_lines += 1

    train_size = int(count_lines * .8)
    test_size = int(count_lines * .1)
    val_size = count_lines - (train_size + test_size)

    print('Train size:', train_size, '; test size:', test_size, '; val size:', val_size)

    with tf.device('/gpu:0'):
        model = tcn_model(num_channels=[1000, 1000, 800, 800, 600, 600, 400, 200, 200, 100, 100, 50, 50])
        model.summary()
        checkpoint = ModelCheckpoint(path_data + 'tsn_weights_1000_to_50.{epoch:03d}-{val_acc:.4f}.hdf5',
                                     monitor='val_acc', verbose=1, mode='auto')

        print('Loading fasttext...')
        fasttext_dict = {}
        with tqdm() as progress:
            with open(path_fasttext_jsonl, 'r') as in_fasttext:
                for line in in_fasttext:
                    embedding = ujson.loads(line)
                    fasttext_dict[embedding['word']] = np.asarray(embedding['embedding'])
                    progress.update()

        print('Training...')
        model.fit_generator(embedded_news_generator(path_news_train, batch_size, fasttext_dict, max_words),
                            steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                            validation_data=embedded_news_generator(path_news_val, batch_size, fasttext_dict,
                                                                    max_words),
                            validation_steps=val_size // batch_size, callbacks=[checkpoint])


def test():
    print('Loading fasttext...')
    cnn_model = tcn_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
