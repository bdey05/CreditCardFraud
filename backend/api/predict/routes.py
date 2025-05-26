from api.predict import bp 

@bp.route('/predict')
def predict() -> str:
    return '<h2>Predict Route</h2>'