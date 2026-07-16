"""FastAPI inference server for Linear Regression.

Implements the shared classical-ML-paper2code2api contract:
    POST /predict   (JSON: {"features": [f1, ..., f8]})  -> {"value": <float>}
    GET  /health                                         -> {status, model_loaded}

Features are RAW, unscaled values in California Housing column order:
    [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]

Run:
    pip install -r requirements.txt
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from infer import ARTIFACT_PATH, FEATURE_NAMES, load_model, predict

app = FastAPI(
    title="classical-ML-paper2code2api · Linear Regression",
    description="California Housing median-value regression (least squares from scratch).",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=f"Raw feature vector in order: {FEATURE_NAMES}",
        examples=[[8.3, 41, 6.98, 1.02, 322, 2.55, 37.88, -122.23]],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": ARTIFACT_PATH.exists()}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> dict:
    try:
        return predict(payload.features)
    except ValueError as exc:
        # Wrong number of features / malformed vector -> client error.
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        # Artifact missing -> model not trained yet.
        raise HTTPException(status_code=503, detail=str(exc))


@app.on_event("startup")
def _warm() -> None:
    # Load params eagerly so the first request isn't slow; tolerate untrained state.
    try:
        load_model()
    except FileNotFoundError:
        pass
