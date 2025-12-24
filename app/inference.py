import pandas as pd
import numpy as np
from .model import model, metadata

DEFAULT_THRESHOLD = metadata["threshold"]

def apply_policy(risk_score: float) -> str:
    if risk_score >= DEFAULT_THRESHOLD:
        return "Rejected"
    elif risk_score >= 0.30:
        return "Manual Review"
    else:
        return "Approved"
    


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # total assets
    df["total_assets"] = (
        df["residential_assets_value"] +
        df["commercial_assets_value"] +
        df["luxury_assets_value"] +
        df["bank_asset_value"]
    )

    # ratios
    df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"]
    df["asset_to_loan_ratio"] = df["total_assets"] / df["loan_amount"]

    return df


def compute_feature_importance(df: pd.DataFrame, top_k: int = 3):
    """
    Menghitung kontribusi fitur berdasarkan:
    |coefficient * input_value|
    """

    # Ambil logistic regression dari pipeline
    logreg = model.named_steps["logreg"]
    coefs = logreg.coef_[0]

    feature_names = df.columns
    values = df.iloc[0].values

    contributions = np.abs(coefs * values)

    importance_df = pd.DataFrame({
        "name": feature_names,
        "contribution": contributions
    })

    # Normalisasi
    importance_df["importance"] = (
        importance_df["contribution"] / importance_df["contribution"].sum()
    )

    # Ambil top features
    top_features = (
        importance_df
        .sort_values("importance", ascending=False)
        .head(top_k)
        [["name", "importance"]]
        .to_dict(orient="records")
    )

    # Round supaya rapi di JSON
    for f in top_features:
        f["importance"] = round(float(f["importance"]), 3)

    return top_features
    

def predict_credit_risk(payload: dict) -> dict:
    df = pd.DataFrame([payload])

    df = preprocess_features(df)

    # Predict probability (Rejected = 1)
    risk_score = model.predict_proba(df)[0, 1]

    decision = apply_policy(risk_score)

    message = (
        "Low risk profile based on provided data."
        if decision == "Approved"
        else "Moderate risk profile. Manual review recommended."
        if decision == "Manual Review"
        else "High risk profile based on provided data."
    )

    top_features = compute_feature_importance(df)

    return {
        "decision": decision,
        "risk_score": round(float(risk_score), 4),
        "features": top_features,
        "message": message,
    }

