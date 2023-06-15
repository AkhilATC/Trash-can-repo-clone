from flask import Flask
from controllers.elk_layer import elk_module
from elasticsearch import Elasticsearch
from config import Config

es = None

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    global es
    es = Elasticsearch([app.config['ELK_URI']])
    app.register_blueprint(elk_module)
    return app
