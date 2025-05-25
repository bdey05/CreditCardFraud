from api.predict import bp 

@bp.route('/predict')
def predict() -> str:
    return '<h2>Test Predict Route</h2>'