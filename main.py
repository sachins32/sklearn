from typing import List
from fastapi import FastAPI
from fastapi.logger import logger
import numpy as np
import pickle

app = FastAPI()
model = pickle.load(open('SVC_classifier.pkl', 'rb'))


@app.get("/")
def read_root():
    return "OK"


@app.post("/predict")
def predict(input: List[float]):
    print(f"Input: {input}")
    print(f"ndarray: {np.array([input]).T}")
    result = model.predict(np.array([input]).T)
    print(f"Result: {result}")
    return result.tolist()

