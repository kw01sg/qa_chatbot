import logging
from logging.config import dictConfig

from config import LOGGING_CONFIG
from flask import Flask

dictConfig(LOGGING_CONFIG)


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('config')
    # app.config.from_pyfile('config.py')

    # init model store
    from flask_app.model import init_model_store
    try:
        init_model_store(app)
    except Exception:
        logging.exception('Unable to init model store. Raising error.')
        raise

    # register blueprints
    from flask_app.views import blueprint
    app.register_blueprint(blueprint)

    return app
