from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import CreditRequest, CreditResponse
from .inference import predict_credit_risk

app = FastAPI(title="Smart Credit Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # untuk dev
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    payload = request.model_dump()
    return predict_credit_risk(payload)
