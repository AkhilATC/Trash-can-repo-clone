from flask import Flask,  jsonify, request
from celery import Celery
from config import Config
from flask_pymongo import PyMongo
from flask_mail import Mail
import os
import celeryconfig
from flask_opensearch import FlaskOpenSearch

app = Flask(__name__)
celery = Celery(__name__)
mongo = PyMongo()
mail = Mail()
opensearch = FlaskOpenSearch()


def make_app():
    env = os.environ.get("ENVIRONMENT", "local")
    app.config.from_object(Config(env=env))

    celery.conf.update(app.config)
    celery.config_from_object(celeryconfig)
    # mongo instance init
    mongo.init_app(app)
    # opensearch instance init
    opensearch.init_app(app)
    # mail service init
    mail.init_app(app)
    from src.alert_controller import alert_interface
    app.register_blueprint(alert_interface)
    
    return app



def make_celery(app):
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
