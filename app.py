from fastapi import FastAPI
import pandas as pd
from pandas import get_dummies
import joblib
from pydantic import BaseModel
app = FastAPI()
model = joblib.load("model.pkl")
bank_columns = joblib.load("features.pkl")
class customerdata(BaseModel):
    age: float
    credit_score: float
    tenure: float
    balance: float
    limite_credito: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float
    cidade: str
    country: str
    gender: str
@app.post("/predict_churn")
def predict_customer_churn(data: customerdata):
    input_dict = data.dict()
    df_input = pd.DataFrame(input_dict)
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=bank_columns, fill_value=0)
    prediction = model.predict(df_input)[0]
    return {"will_churn": int(prediction)}
