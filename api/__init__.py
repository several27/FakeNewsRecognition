import logging
from datetime import datetime
from typing import Union, List

import gensim
import newspaper
import numpy as np
from flask import Flask, jsonify, abort, request
from flask.json import JSONEncoder
from flask_mail import Mail
from keras.engine import Input
from keras.layers import Bidirectional, LSTM, Dense, Dropout, Conv1D, Activation, Flatten, Add
from keras.models import Model
from peewee import DoesNotExist
from voluptuous import Schema, Invalid, Required, Optional

from api.database import WordEmbedding

app = Flask(__name__)

mail = Mail(app)

logger = logging.getLogger('app')
handler = logging.FileHandler('/src/api/index.log')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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


def embed(content: str) -> np.ndarray:
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
        try:
            word_embedding = WordEmbedding.get(
                (WordEmbedding.version == 1) & (WordEmbedding.word == word))  # type: WordEmbedding
        except DoesNotExist:
            continue

        embedding[i] = word_embedding.embedding
        i += 1

        if i >= max_words:
            break

    return embedding


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, datetime):
                return obj.isoformat()

            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)

        return JSONEncoder.default(self, obj)


app.json_encoder = CustomJSONEncoder

welcome_message = 'Welcome in the Fake News Recognition API. Please see ' \
                  'https://github.com/several27/FakeNewsRecognition to learn more!'


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': welcome_message
    })


@app.route('/v1', methods=['GET'])
def home_v1():
    return jsonify({
        'message': welcome_message
    })


model_all = tcn_all_model(num_channels=[1000] * 4 + [100])
model_all.load_weights('/src/api/tsn_weights_word_cnn_all.002-0.8443.hdf5')
print('Models loaded')


@app.route('/v1/predict', methods=['POST'])
def predict():
    data = request.get_json()

    validate_schema(data, Schema({
        Required('url'): str,
        Optional('title'): Optional(str),
        Optional('content'): Optional(str)
    }))

    if 'title' not in data or 'content' not in data or data['title'] is None or data['content'] is None:
        article = newspaper.Article(data['url'])
        article.download()
        article.parse()
        title = article.title
        content = article.text
    else:
        title = data['title']
        content = data['content']

    embedding = embed(title + content)

    model = bilstm_model()
    model.load_weights('/src/api/bilstm_weights.010-0.9848.hdf5')
    binary_prediction = float(model.predict(embedding.reshape((1, input_shape[0], input_shape[1])))[0])

    # model_all = bilstm_all_model(units=[128, 64, 32], hidden_dims=18)
    # model_all.load_weights('/src/api/bilstm_all_1_weights.002-0.8368.hdf5')
    predictions = [float(v) for v in model_all.predict(embedding.reshape((1, input_shape[0], input_shape[1])))[0]]

    return jsonify({
        'status': 200,
        'data': {
            'fake': 1 - binary_prediction,
            'prediction': 'fake' if binary_prediction <= 0.5 else 'true',
            'classes': dict([(l, predictions[i]) for i, l in enumerate(news_labels)])
        }
    })


def validate_schema(data, schema):
    try:
        Schema(schema)(data)
    except Invalid as e:
        return abort(400, str(e))


def error_handler(code, message):
    response = jsonify({
        'status': code,
        'message': message
    })
    response.status_code = code

    return response


@app.errorhandler(500)
def error_handler_500(error):
    return error_handler(500, error.description)


@app.errorhandler(404)
def error_handler_404(error):
    return error_handler(404, error.description)


@app.errorhandler(403)
def error_handler_403(error):
    return error_handler(403, error.description)


@app.errorhandler(401)
def error_handler_401(error):
    return error_handler(401, error.description)


@app.errorhandler(400)
def error_handler_400(error):
    return error_handler(400, error.description)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    # context = SSL.Context(SSL.SSLv23_METHOD)
    # context.use_privatekey_file('server/SSL/api-fusemind.key')
    # context.use_certificate_file('server/SSL/api-fusemind.crt')
    # context = ('server/SSL/api-fusemind.crt', 'server/SSL/api-fusemind.key')
    app.run(host='0.0.0.0', port=8080)
