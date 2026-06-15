"""
FraudPredictor — Chargement et inférence des modèles entraînés.
Ensemble : LSTM + XGBoost (Autoencoder conservé comme référence).
"""

import numpy as np
import joblib
import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


PAYMENT_MAP = {
    "Credit Card": 0, "Debit Card": 1, "PayPal": 2,
    "Bank Transfer": 3, "Cryptocurrency": 4
}
CATEGORY_MAP = {
    "Electronics": 0, "Clothing": 1, "Books": 2,
    "Home & Garden": 3, "Sports": 4, "Toys": 5, "Other": 6
}
DEVICE_MAP = {"Desktop": 0, "Mobile": 1, "Tablet": 2}

SEQ_LEN = 10  # Longueur des séquences LSTM


class FraudPredictor:
    """
    Charge les modèles sauvegardés et effectue les prédictions.
    Ensemble final : LSTM + XGBoost.
    Autoencoder disponible comme score de référence.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir     = models_dir
        self.xgb_model      = None
        self.autoencoder    = None
        self.lstm_model     = None
        self.scaler         = None
        self.feature_names  = None
        self.ae_threshold   = 0.05
        self.ensemble_cfg   = {"w_lstm": 0.35, "w_xgb": 0.65}
        self._load_models()

    def _load_models(self):
        """Charge tous les modèles disponibles."""
        # XGBoost
        xgb_path = os.path.join(self.models_dir, "xgboost_model.pkl")
        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            logger.info("✅ XGBoost chargé")

        # Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler chargé")

        # Keras models (Autoencoder + LSTM)
        try:
            from tensorflow import keras

            ae_path = os.path.join(self.models_dir, "autoencoder.h5")
            if os.path.exists(ae_path):
                self.autoencoder = keras.models.load_model(ae_path)
                logger.info("✅ Autoencoder chargé")

            lstm_path = os.path.join(self.models_dir, "lstm_model.h5")
            if os.path.exists(lstm_path):
                self.lstm_model = keras.models.load_model(lstm_path)
                logger.info("✅ LSTM chargé")

        except Exception as e:
            logger.warning(f"⚠️ Keras models non chargés: {e}")

        # Threshold AE
        thresh_path = os.path.join(self.models_dir, "ae_threshold.npy")
        if os.path.exists(thresh_path):
            self.ae_threshold = float(np.load(thresh_path))

        # Feature names
        feat_path = os.path.join(self.models_dir, "feature_names.json")
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.feature_names = json.load(f)

        # Ensemble config
        ens_path = os.path.join(self.models_dir, "ensemble_config.json")
        if os.path.exists(ens_path):
            with open(ens_path) as f:
                self.ensemble_cfg = json.load(f)

    def get_models_status(self) -> dict:
        return {
            "xgboost":     self.xgb_model is not None,
            "lstm":        self.lstm_model is not None,
            "autoencoder": self.autoencoder is not None,
            "scaler":      self.scaler is not None,
            "feature_names": self.feature_names is not None,
        }

    def _build_feature_vector(self, tx: dict) -> np.ndarray:
        """Construit le vecteur de features à partir des inputs API."""
        amount    = tx["transaction_amount"]
        quantity  = tx["quantity"]
        cust_age  = tx["customer_age"]
        acc_age   = tx["account_age_days"]
        tx_hour   = tx["transaction_hour"]
        pay_enc   = PAYMENT_MAP.get(tx["payment_method"], 0)
        cat_enc   = CATEGORY_MAP.get(tx["product_category"], 0)
        dev_enc   = DEVICE_MAP.get(tx["device_used"], 0)
        same_addr = int(tx["same_address"])

        # Features dérivées
        amount_log       = np.log1p(amount)
        amount_per_unit  = amount / (quantity + 1)
        is_night         = int(tx.get("is_night") or (tx_hour >= 22 or tx_hour <= 5))
        is_weekend       = int(tx.get("is_weekend") or False)

        # Features client
        cust_tx_count    = tx.get("customer_tx_count") or 1
        cust_avg         = tx.get("customer_avg_amount") or amount
        cust_std         = abs(amount - cust_avg) * 0.5
        amount_zscore    = (amount - cust_avg) / (cust_std + 1e-8)
        amount_vs_median = amount / (cust_avg + 1)

        feature_vector = np.array([
            amount, quantity, cust_age, acc_age, tx_hour,
            amount_log, amount_per_unit, amount_vs_median,
            is_weekend, is_night,
            0, 0,          # tx_day_of_week, tx_month
            same_addr,
            cust_tx_count, cust_avg, cust_std, amount_zscore,
            pay_enc, cat_enc, dev_enc
        ], dtype=np.float32)

        return feature_vector.reshape(1, -1)

    def _build_lstm_sequence(self, feature_vec_scaled: np.ndarray) -> np.ndarray:
        """
        Construit une séquence LSTM pour une transaction isolée.
        La transaction est placée en dernière position, le reste est zéro-paddé.
        """
        n_features = feature_vec_scaled.shape[1]
        seq = np.zeros((1, SEQ_LEN, n_features), dtype=np.float32)
        seq[0, -1, :] = feature_vec_scaled[0]
        return seq

    def _get_risk_factors(self, tx: dict, xgb_score: float, lstm_score: float) -> list:
        """Identifie les principaux facteurs de risque."""
        factors = []
        if tx["transaction_amount"] > 1000:
            factors.append(f"Montant élevé ({tx['transaction_amount']:.0f}€)")
        if not tx["same_address"]:
            factors.append("Adresse livraison ≠ facturation")
        if tx["account_age_days"] < 30:
            factors.append(f"Compte récent ({tx['account_age_days']} jours)")
        if tx["payment_method"] == "Cryptocurrency":
            factors.append("Paiement en cryptomonnaie")
        if tx.get("is_night") or (tx["transaction_hour"] >= 22 or tx["transaction_hour"] <= 5):
            factors.append(f"Transaction nocturne ({tx['transaction_hour']}h)")
        if tx["quantity"] > 5:
            factors.append(f"Quantité élevée ({tx['quantity']} unités)")
        if tx.get("customer_tx_count", 99) == 1:
            factors.append("Première transaction du client")
        if lstm_score > 0.7:
            factors.append("Comportement séquentiel suspect (LSTM)")
        cust_avg = tx.get("customer_avg_amount")
        if cust_avg and tx["transaction_amount"] > cust_avg * 3:
            factors.append(f"Montant 3x supérieur à la moyenne ({cust_avg:.0f}€)")
        return factors[:5] if factors else ["Aucun facteur de risque majeur identifié"]

    def predict(self, tx: dict) -> dict:
        """
        Prédiction principale via ensemble LSTM + XGBoost.
        Fonctionne même si certains modèles ne sont pas chargés (fallback heuristique).
        """
        feature_vec = self._build_feature_vector(tx)

        # ── Scaling ──────────────────────────────────────────────
        if self.scaler:
            feat_scaled = self.scaler.transform(feature_vec)
        else:
            feat_scaled = feature_vec

        # ── XGBoost ──────────────────────────────────────────────
        if self.xgb_model and self.scaler:
            try:
                xgb_score = float(self.xgb_model.predict_proba(feat_scaled)[0, 1])
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                xgb_score = self._heuristic_score(tx)
        else:
            xgb_score = self._heuristic_score(tx)

        # ── LSTM ─────────────────────────────────────────────────
        if self.lstm_model and self.scaler:
            try:
                seq = self._build_lstm_sequence(feat_scaled)
                lstm_score = float(self.lstm_model.predict(seq, verbose=0)[0, 0])
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                lstm_score = xgb_score * 0.9
        else:
            lstm_score = xgb_score * 0.9

        # ── Autoencoder (référence, non inclus dans l'ensemble) ──
        if self.autoencoder and self.scaler:
            try:
                recon = self.autoencoder.predict(feat_scaled, verbose=0)
                recon_err = float(np.mean(np.square(feat_scaled - recon)))
                ae_raw = recon_err / (self.ae_threshold * 3)
                ae_score = min(ae_raw, 1.0)
            except Exception as e:
                logger.warning(f"Autoencoder prediction failed: {e}")
                ae_score = xgb_score * 0.8
        else:
            ae_score = xgb_score * 0.8

        # ── Ensemble : LSTM + XGBoost ─────────────────────────────
        w_lstm = self.ensemble_cfg.get("w_lstm", 0.35)
        w_xgb  = self.ensemble_cfg.get("w_xgb", 0.65)
        ensemble_score = w_lstm * lstm_score + w_xgb * xgb_score

        is_fraud   = ensemble_score > 0.5
        confidence = abs(ensemble_score - 0.5) * 2

        return {
            "is_fraud":         is_fraud,
            "xgboost_score":    xgb_score,
            "lstm_score":       lstm_score,
            "autoencoder_score": ae_score,
            "ensemble_score":   ensemble_score,
            "confidence":       confidence,
            "top_risk_factors": self._get_risk_factors(tx, xgb_score, lstm_score)
        }

    def _heuristic_score(self, tx: dict) -> float:
        """Score heuristique de fallback quand les modèles ne sont pas chargés."""
        score = 0.1
        if tx["transaction_amount"] > 500:          score += 0.15
        if tx["transaction_amount"] > 2000:         score += 0.20
        if not tx["same_address"]:                  score += 0.20
        if tx["account_age_days"] < 30:             score += 0.20
        if tx["payment_method"] == "Cryptocurrency": score += 0.15
        hour = tx["transaction_hour"]
        if hour >= 22 or hour <= 4:                 score += 0.10
        if tx["quantity"] > 10:                     score += 0.10
        return min(score, 0.99)
