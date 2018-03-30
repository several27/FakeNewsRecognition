import tensorflow as tf
from gensim.models.fasttext import FastText
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input, Dense, Bidirectional, LSTM
from keras.models import Model
from tqdm import tqdm

from machines.data_generator import embedded_news_generator, path_data, path_news_train, path_news_val, path_fasttext, \
    path_news_shuffled

max_words = 300
input_shape = max_words, 100

batch_size = 64
epochs = 10


def lstm_model(units=64, dropout=(0.5,), hidden_dims=17):
    model_input = Input(shape=input_shape)
    lstm = LSTM(units)(model_input)

    z = Dropout(dropout[0])(lstm)
    z = Dense(hidden_dims, activation='relu')(z)
    model_output = Dense(1, activation='sigmoid')(z)

    model = Model(model_input, model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train():
    print('Loading fasttext...')
    fasttext = FastText.load_fasttext_format(path_fasttext)
    fasttext_dict = {}
    for word in tqdm(fasttext.wv.vocab):
        fasttext_dict[word] = fasttext[word]

    del fasttext

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
        cnn_model = lstm_model(units=128, hidden_dims=32)
        checkpoint = ModelCheckpoint(path_data + 'lstm_weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                                     verbose=1, mode='auto')
        cnn_model.fit_generator(embedded_news_generator(path_news_train, batch_size, fasttext_dict, max_words),
                                steps_per_epoch=train_size // batch_size, epochs=epochs, verbose=1,
                                validation_data=embedded_news_generator(path_news_val, batch_size, fasttext_dict,
                                                                        max_words),
                                validation_steps=val_size // batch_size, callbacks=[checkpoint])


def test():
    print('Loading fasttext...')
    cnn_model = lstm_model()
    cnn_model.load_weights(path_data + 'cnn_deep_weights.000-0.4900.hdf5')


if __name__ == '__main__':
    train()
