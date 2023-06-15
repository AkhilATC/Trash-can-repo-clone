from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///annual_data.sqlite3"
db = SQLAlchemy(app)


def create_app():
    # return flask app object
    from controller.data_interface import data_layer
    app.register_blueprint(data_layer)
    return app

