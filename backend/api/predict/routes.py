from api.predict import bp
from flask import jsonify, Response, request
import pandas as pd
import numpy as np
import joblib 

@bp.route("/predict", methods=["POST"])
def predict() -> Response:
    file = request.files['file']
    df = pd.read_csv(file, encoding='utf-8')
    X = df.loc[:, df.columns != 'Class']
    final_model = joblib.load("./model/final_model.joblib")
    predictions = final_model.predict(X)
    fraud_count = np.sum(predictions == 1)
    return jsonify({"predicted_fraud_count": int(fraud_count), "total_transactions_processed": len(df)})
