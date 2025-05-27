from flask import Blueprint

bp = Blueprint("predict", __name__)

from api.predict import routes
