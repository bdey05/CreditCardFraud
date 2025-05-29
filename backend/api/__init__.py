from flask import Flask
from config import Config
from flask_cors import CORS 



def create_app(config_class=Config):
    app = Flask(__name__)
    CORS(app) 
    
    app.config.from_object(config_class)


    from api.predict import bp

    app.register_blueprint(bp)

    return app
