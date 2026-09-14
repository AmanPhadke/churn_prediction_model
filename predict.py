from fastapi import FastAPI
from pydantic import BaseModel

import pickle
# from flask import Flask
# from flask import request
# from flask import jsonify


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = FastAPI()

class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure: int
    monthlycharges: float
    totalcharges: float

@app.post("/predict")
def predict(customer: Customer):
    customer = customer.model_dump()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    churn = (y_pred >= 0.5)

    result = {
        'churn': bool(churn),
        'churn_probability': float(y_pred)
    }

    return (result)
    

# app = Flask('churn')

# @app.route('/predict', methods=['POST'])
# def predict():
#     customer = request.get_json()

#     X = dv.transform([customer])
#     y_pred = model.predict_proba(X)[:,1]
#     churn = (y_pred >= 0.5)

#     result = {
#         'churn_probability': float(y_pred),
#         'churn': bool(churn)
#     }

#     return jsonify(result)

# if __name__ == "__main__":
#      app.run(debug=True, host='0.0.0.0', port = 9696)