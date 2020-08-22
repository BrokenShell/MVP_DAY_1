from flask import Flask, jsonify
from app.model import PredictionBot

API = Flask(__name__)
PB = PredictionBot()


@API.route('/')
def index():
    return jsonify("API online!")


@API.route('/<user_input>')
def search(user_input):
    return jsonify(PB.predict(user_input))


if __name__ == '__main__':
    API.run(debug=True)
