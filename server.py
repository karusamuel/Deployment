from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app =  FastAPI()

model = joblib.load("./model/model.joblib")

class Person(BaseModel):
    
    name:str

class Features(BaseModel):
    
    # ["displacement","weight","horsepower","acceleration"]
    displacement:float
    weight:float
    horsepower:float
    acceleration:float


@app.get("/")
def hello():
    
    return {"message":"Hello"}

@app.post("/name")
def hello_name(person:Person):
    name = "example"
    return {"message":f"hello {person.name}"}

@app.post("/predict")
def predict(features:Features):
    df = pd.DataFrame([features.model_dump()])
    y_pred = model.predict(df)
    return {"prediction":f"predicted MPG {y_pred.tolist()[0]}"}



