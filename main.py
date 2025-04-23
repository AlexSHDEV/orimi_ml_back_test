from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import os
from fastapi.responses import JSONResponse
import ml
import json
import catboost

app = FastAPI()

total_queries = 0
successful_queries = 0


model = None
MODEL_PATH = "model.pkl"
MODEL_NAME = "catboost"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model '{MODEL_NAME}' loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/ping")
async def ping(request: Request):
    global total_queries, successful_queries
    total_queries += 1
    successful_queries += 1

    return {
        "status": "ok",
        "total_queries": total_queries,
        "successful_queries": successful_queries
    }


@app.post("/inference")
async def inference(request: Request):
    global total_queries, successful_queries

    total_queries += 1

    body_data = {}
    try:
        body_bytes = await request.body()
        if body_bytes:
            json_str = body_bytes.decode('utf-8')
            json_dict = json.loads(json_str)
            input = ml.PromoData(**json_dict)
    except json.JSONDecodeError:
        pass

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input = ml.preproc(input)
        prediction = model.predict([list(input.values)])[0]
        successful_queries += 1

        return {
            "prediction": int(prediction),
            "model": MODEL_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)