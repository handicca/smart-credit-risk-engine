from pydantic import BaseModel
from typing import List


class CreditRequest(BaseModel):
    no_of_dependents: int
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


class FeatureImportance(BaseModel):
    name: str
    importance: float


class CreditResponse(BaseModel):
    decision: str
    risk_score: float
    features: List[FeatureImportance]
    message: str