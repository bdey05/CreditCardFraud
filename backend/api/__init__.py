from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)

    """with app.app_context():
        try:
            db.create_all()
        except:
            app.logger.error('Failed to create database tables')"""

    from api.predict import bp

    app.register_blueprint(bp)

    return app
