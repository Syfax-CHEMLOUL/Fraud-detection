"""
🛡️ Fraud Detection API — FastAPI
================================
Détection de fraude en temps réel via ensemble LSTM + XGBoost.

Usage:
    uvicorn src.api.main:app --reload --port 8000
    Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import numpy as np
import time
import logging

from src.api.predictor import FraudPredictor

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="🛡️ Fraud Detection API",
    description="""
## Fraud Detection in Finance

API de détection de fraude en temps réel utilisant un ensemble de modèles ML :
- **LSTM** : Analyse comportementale séquentielle (35%)
- **XGBoost** : Gradient Boosting sur features de transaction (65%)
- **Ensemble** : Combinaison pondérée LSTM + XGBoost
- **Autoencoder** : Disponible comme score de référence (non inclus dans l'ensemble)

### Comment utiliser ?
1. POST `/predict` avec les informations de transaction
2. Récupérez le score de risque et la décision
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Charger le predictor au démarrage ───────────────────────────
predictor = FraudPredictor(models_dir="models")


# ── Schémas Pydantic ────────────────────────────────────────────
class TransactionInput(BaseModel):
    """Informations de transaction pour la prédiction."""

    # Montant et quantité
    transaction_amount: float = Field(..., gt=0, description="Montant de la transaction (€)", example=250.0)
    quantity: int = Field(..., gt=0, le=1000, description="Nombre d'articles", example=2)

    # Client
    customer_age: int = Field(..., ge=18, le=100, description="Âge du client", example=35)
    account_age_days: int = Field(..., ge=0, description="Ancienneté du compte en jours", example=365)

    # Transaction
    transaction_hour: int = Field(..., ge=0, le=23, description="Heure de la transaction (0-23)", example=14)
    payment_method: Literal["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cryptocurrency"] = Field(
        ..., description="Méthode de paiement", example="Credit Card"
    )
    product_category: Literal["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Toys", "Other"] = Field(
        ..., description="Catégorie du produit", example="Electronics"
    )
    device_used: Literal["Desktop", "Mobile", "Tablet"] = Field(
        ..., description="Appareil utilisé", example="Mobile"
    )

    # Adresses
    same_address: bool = Field(..., description="Adresse livraison = adresse facturation ?", example=True)

    # Features contextuelles (optionnelles)
    is_weekend: Optional[bool] = Field(None, description="Transaction le week-end ?", example=False)
    is_night: Optional[bool] = Field(None, description="Transaction la nuit (22h-5h) ?", example=False)
    customer_tx_count: Optional[int] = Field(None, ge=1, description="Nombre total de transactions du client", example=12)
    customer_avg_amount: Optional[float] = Field(None, description="Montant moyen habituel du client (€)", example=150.0)

    @validator('transaction_amount')
    def amount_reasonable(cls, v):
        if v > 100_000:
            raise ValueError("Montant trop élevé (max 100,000)")
        return v


class PredictionResponse(BaseModel):
    """Résultat de la prédiction."""
    transaction_id: Optional[str]
    is_fraud: bool
    risk_score: float = Field(..., description="Score de risque [0-1]")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(..., description="Confiance de la prédiction [0-1]")

    # Scores par modèle
    xgboost_score: float
    lstm_score: float
    autoencoder_score: float  # Référence uniquement
    ensemble_score: float

    # Explications
    top_risk_factors: list[str]
    recommendation: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    ensemble: str
    version: str


# ── Routes ───────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "🛡️ Fraud Detection API v2.0",
        "ensemble": "LSTM (35%) + XGBoost (65%)",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Vérifier l'état de l'API et des modèles."""
    return HealthResponse(
        status="healthy",
        models_loaded=predictor.get_models_status(),
        ensemble="LSTM (35%) + XGBoost (65%)",
        version="2.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(transaction: TransactionInput, transaction_id: Optional[str] = None):
    """
    ## Prédire si une transaction est frauduleuse

    Soumettez les informations d'une transaction et recevez :
    - **is_fraud** : Décision binaire
    - **risk_score** : Score de risque de 0 (sûr) à 1 (fraude certaine)
    - **risk_level** : LOW / MEDIUM / HIGH / CRITICAL
    - **top_risk_factors** : Principaux facteurs de risque détectés

    ### Ensemble
    Score final = 0.35 × LSTM + 0.65 × XGBoost
    """
    start = time.time()

    try:
        result = predictor.predict(transaction.dict())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

    elapsed_ms = (time.time() - start) * 1000

    # Niveau de risque
    score = result["ensemble_score"]
    if score < 0.3:
        risk_level = "LOW"
        recommendation = "✅ Transaction approuvée. Aucune action requise."
    elif score < 0.5:
        risk_level = "MEDIUM"
        recommendation = "⚠️ Surveillance recommandée. Vérification manuelle possible."
    elif score < 0.75:
        risk_level = "HIGH"
        recommendation = "🔴 Transaction à risque élevé. Vérification manuelle requise."
    else:
        risk_level = "CRITICAL"
        recommendation = "🚨 FRAUDE PROBABLE. Bloquer et alerter le client immédiatement."

    logger.info(f"[{transaction_id or 'N/A'}] Score={score:.3f} | Level={risk_level} | {elapsed_ms:.0f}ms")

    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=result["is_fraud"],
        risk_score=round(score, 4),
        risk_level=risk_level,
        confidence=round(result["confidence"], 4),
        xgboost_score=round(result["xgboost_score"], 4),
        lstm_score=round(result["lstm_score"], 4),
        autoencoder_score=round(result["autoencoder_score"], 4),
        ensemble_score=round(score, 4),
        top_risk_factors=result["top_risk_factors"],
        recommendation=recommendation,
        processing_time_ms=round(elapsed_ms, 2)
    )


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(transactions: list[TransactionInput]):
    """
    ## Prédiction en lot (max 100 transactions)
    """
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 transactions par lot")

    results = []
    for i, tx in enumerate(transactions):
        try:
            result = predictor.predict(tx.dict())
            results.append({"index": i, "status": "ok", **result})
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})

    return {
        "total": len(transactions),
        "frauds_detected": sum(1 for r in results if r.get("is_fraud")),
        "results": results
    }


@app.get("/examples", tags=["Info"])
def get_examples():
    """Exemples de transactions pour tester l'API."""
    return {
        "transaction_normale": {
            "transaction_amount": 89.99,
            "quantity": 1,
            "customer_age": 32,
            "account_age_days": 730,
            "transaction_hour": 15,
            "payment_method": "Credit Card",
            "product_category": "Clothing",
            "device_used": "Desktop",
            "same_address": True,
            "is_weekend": False,
            "is_night": False,
            "customer_tx_count": 25,
            "customer_avg_amount": 95.0
        },
        "transaction_suspecte": {
            "transaction_amount": 4999.99,
            "quantity": 10,
            "customer_age": 22,
            "account_age_days": 3,
            "transaction_hour": 3,
            "payment_method": "Cryptocurrency",
            "product_category": "Electronics",
            "device_used": "Mobile",
            "same_address": False,
            "is_weekend": True,
            "is_night": True,
            "customer_tx_count": 1,
            "customer_avg_amount": 4999.99
        }
    }
