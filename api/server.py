import os
import logging
from datetime import datetime

import boto3
import newspaper
from flask import Flask, jsonify, abort, request, send_from_directory
from flask.json import JSONEncoder
from flask_mail import Mail
from voluptuous import Schema, Invalid, Required, Optional

from api.helpers import bilstm_model, embed, input_shape, news_labels, tcn_all_model

app = Flask(__name__)

mail = Mail(app)

logger = logging.getLogger('app')
handler = logging.FileHandler('/src/api/index.log')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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


@app.route('/api/', methods=['GET'])
def home():
    return jsonify({
        'message': welcome_message
    })


@app.route('/api/v1', methods=['GET'])
def home_v1():
    return jsonify({
        'message': welcome_message
    })


path_binary_model = '/src/data/binary_model.hdf5'
path_multiclass_model = '/src/data/multiclass_model.hdf5'
if not os.path.isfile(path_binary_model) or not os.path.isfile(path_multiclass_model):
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket('fake-news-recognition')
    s3_bucket.download_file('bilstm_weights.010-0.9848.hdf5', path_binary_model)
    s3_bucket.download_file('tsn_weights_word_cnn_all.002-0.8443.hdf5', path_multiclass_model)

model_all = tcn_all_model(num_channels=[1000] * 4 + [100])
model_all.load_weights(path_multiclass_model)
print('Models loaded')


@app.route('/api/v1/predict', methods=['POST'])
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
    model.load_weights(path_binary_model)
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
