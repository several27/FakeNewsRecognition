from datetime import datetime
import logging

from flask import Flask, jsonify, abort, g
from flask.json import JSONEncoder
from flask_cors import CORS
from flask_mail import Mail
from flask_sslify import SSLify

app = Flask(__name__)

sslify = SSLify(app)

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
