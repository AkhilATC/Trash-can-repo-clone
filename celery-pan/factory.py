from flask import Flask
from celery import Celery
from config import Config

app = Flask(__name__)
celery = Celery(__name__)


def make_app():
    app.config.from_object(Config)
    from src.alert_controller import alert_interface
    app.register_blueprint(alert_interface)
    celery.conf.update(app.config)
    return app


def make_celery(app):
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery