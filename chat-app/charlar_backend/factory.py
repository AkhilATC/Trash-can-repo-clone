from os import path
import sys
print(path.abspath(__file__))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from flask import Flask
from flask_pymongo import PyMongo
from flask_cors import CORS, cross_origin
import config
from flask_jwt_extended import JWTManager

mongo_db = PyMongo()
app = Flask(__name__)
CORS(app)
jwt = JWTManager(app)


def setup_flask_app():
    app.config.from_object(config.Config)
    from controllers.auth_controller import auth_module
    from controllers.chat_controller import chat_module
    app.register_blueprint(auth_module)
    app.register_blueprint(chat_module)
    mongo_db.init_app(app)
    return app