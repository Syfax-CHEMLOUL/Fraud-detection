"""
Tests pour l'API de détection de fraude.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

NORMAL_TX = {
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
}

FRAUD_TX = {
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


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "Fraud Detection" in data["message"]
    assert "ensemble" in data


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "lstm" in data["models_loaded"]
    assert "xgboost" in data["models_loaded"]


def test_predict_normal_transaction():
    resp = client.post("/predict", json=NORMAL_TX)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert "risk_score" in data
    assert "lstm_score" in data
    assert "xgboost_score" in data
    assert "autoencoder_score" in data
    assert 0 <= data["risk_score"] <= 1
    assert 0 <= data["lstm_score"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(data["top_risk_factors"]) > 0


def test_predict_suspicious_transaction():
    resp = client.post("/predict", json=FRAUD_TX)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_score"] > 0.3, "Transaction suspecte doit avoir un score élevé"
    assert data["risk_level"] in ["HIGH", "CRITICAL"]


def test_predict_with_transaction_id():
    resp = client.post("/predict?transaction_id=TX-001", json=NORMAL_TX)
    assert resp.status_code == 200
    assert resp.json()["transaction_id"] == "TX-001"


def test_predict_invalid_amount():
    tx = NORMAL_TX.copy()
    tx["transaction_amount"] = -50
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 422


def test_predict_invalid_hour():
    tx = NORMAL_TX.copy()
    tx["transaction_hour"] = 25
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 422


def test_batch_predict():
    resp = client.post("/predict/batch", json=[NORMAL_TX, FRAUD_TX])
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert "frauds_detected" in data
    assert len(data["results"]) == 2


def test_batch_limit():
    txs = [NORMAL_TX] * 101
    resp = client.post("/predict/batch", json=txs)
    assert resp.status_code == 400


def test_examples_endpoint():
    resp = client.get("/examples")
    assert resp.status_code == 200
    data = resp.json()
    assert "transaction_normale" in data
    assert "transaction_suspecte" in data


def test_response_time():
    import time
    start = time.time()
    resp = client.post("/predict", json=NORMAL_TX)
    elapsed = (time.time() - start) * 1000
    assert resp.status_code == 200
    assert elapsed < 2000, f"Trop lent: {elapsed:.0f}ms"
