from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import numpy as np
import pandas as pd
from src.End_To_End_Wine_Prediction.pipeline.prediction import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...),
):
    try:
        data = [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]
        data = np.array(data).reshape(1, 11)

        obj = PredictionPipeline()
        predict = obj.predict(data)

        return templates.TemplateResponse(
            "results.html", {"request": request, "prediction": str(predict)}
        )
    except Exception as e:
        print("The Exception message is: ", e)
        return "Something went wrong"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)