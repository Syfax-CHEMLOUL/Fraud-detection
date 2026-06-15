# 🛡️ Fraud Detection in Finance

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Système de détection de fraude en temps réel combinant **LSTM** et **XGBoost** dans un ensemble hybride, exposé via une API REST prête pour la production.

---

## 📋 Table des matières

- [Aperçu du projet](#-aperçu-du-projet)
- [Dataset](#-dataset)
- [Architecture des modèles](#-architecture-des-modèles)
- [Feature Engineering](#-feature-engineering)
- [Résultats](#-résultats)
- [API REST](#-api-rest)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)

---

## 🎯 Aperçu du projet

Pipeline complet de détection de fraude pour les transactions e-commerce, de la collecte des données jusqu'au déploiement en API. Trois approches complémentaires sont entraînées ; deux sont combinées dans l'ensemble final :

| Modèle | Approche | Rôle |
|--------|----------|------|
| **Autoencoder** | Non supervisée | Détection d'anomalies — score de référence |
| **LSTM + Attention** | Séquentielle | Ensemble final (35%) |
| **XGBoost** | Supervisée | Ensemble final (65%) |
| **Ensemble** | Hybride | `0.35 × LSTM + 0.65 × XGBoost` |

> **Dataset réduit** : Le notebook utilise un sous-échantillon stratifié de **15 000 transactions** pour un entraînement rapide (< 10 min sur CPU).

---

## 📊 Dataset

**Source :** [Fraudulent E-Commerce Transactions — Kaggle](https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions)

- Dataset complet : ~150 000 transactions e-commerce
- **Sous-échantillon utilisé : 15 000 lignes** (stratifié, taux de fraude ~10% préservé)
- 15+ features : montant, méthode de paiement, appareil, localisation, etc.

---

## 🧠 Architecture des modèles

### 1. Autoencoder (Référence)

```
Input (20 features)
  → Dense(64) + BatchNorm + Dropout(0.3)
  → Dense(32) + BatchNorm + Dropout(0.2)
  → Latent Space (4 dims)
  → Dense(32) + BatchNorm + Dropout(0.2)
  → Dense(64) + BatchNorm
  → Output (20 features)
```

Entraîné uniquement sur les transactions normales. Erreur de reconstruction élevée → comportement anormal. Utilisé comme score de référence, non inclus dans l'ensemble final.

### 2. LSTM avec Attention (Ensemble — 35%)

```
Input sequences (10 transactions × 20 features)
  → LSTM(64, return_sequences=True) + Dropout(0.3)
  → LSTM(32, return_sequences=True) + Dropout(0.3)
  → Self-Attention Layer
  → Dense(32) + BatchNorm + Dropout(0.3)
  → Sigmoid Output
```

Capture les patterns comportementaux sur les 10 dernières transactions du client. Pour une transaction isolée, la séquence est zero-paddée.

### 3. XGBoost (Ensemble — 65%)

- 200 estimateurs, `max_depth=6`
- `scale_pos_weight` pour gérer le déséquilibre de classes
- SMOTE (`k_neighbors=3`) pour l'oversampling des fraudes
- SHAP values pour l'explicabilité

### 4. Ensemble Pondéré

```
Score_final = 0.35 × Score_LSTM + 0.65 × Score_XGBoost
```

---

## ⚙️ Feature Engineering

20 features :

**Brutes :** Montant, Quantité, Âge client, Ancienneté compte, Heure

**Dérivées :**
- `amount_log` — Montant en échelle logarithmique
- `amount_per_unit` — Montant par article
- `amount_zscore` — Écart par rapport à la moyenne client
- `tx_is_night` — Transaction entre 22h et 5h
- `tx_is_weekend` — Transaction le week-end
- `same_address` — Adresse livraison = facturation
- `customer_tx_count` — Nombre de transactions du client
- `amount_vs_cat_median` — Montant vs médiane de la catégorie

**Encodées :** Méthode de paiement, Catégorie produit, Appareil

---

## 📈 Résultats

| Modèle | AUC-ROC | Avg Precision | F1 Score |
|--------|---------|---------------|----------|
| Autoencoder | ~0.80 | ~0.62 | ~0.68 |
| XGBoost | ~0.93 | ~0.85 | ~0.84 |
| LSTM | ~0.88 | ~0.78 | ~0.80 |
| **Ensemble (LSTM+XGB)** | **~0.94** | **~0.87** | **~0.86** |

*Résultats indicatifs sur sous-échantillon de 15 000 lignes — varient selon le run.*

---

## 🚀 API REST

### `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 4999.99,
    "quantity": 10,
    "customer_age": 22,
    "account_age_days": 3,
    "transaction_hour": 3,
    "payment_method": "Cryptocurrency",
    "product_category": "Electronics",
    "device_used": "Mobile",
    "same_address": false,
    "is_night": true
  }'
```

**Réponse :**
```json
{
  "is_fraud": true,
  "risk_score": 0.8712,
  "risk_level": "CRITICAL",
  "confidence": 0.7424,
  "xgboost_score": 0.9102,
  "lstm_score": 0.7843,
  "autoencoder_score": 0.8431,
  "ensemble_score": 0.8712,
  "top_risk_factors": [
    "Montant élevé (4999€)",
    "Compte récent (3 jours)",
    "Paiement en cryptomonnaie",
    "Transaction nocturne (3h)",
    "Adresse livraison ≠ facturation"
  ],
  "recommendation": "🚨 FRAUDE PROBABLE. Bloquer et alerter le client immédiatement.",
  "processing_time_ms": 18.3
}
```

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info de l'API |
| `/health` | GET | État des modèles |
| `/predict` | POST | Prédiction simple |
| `/predict/batch` | POST | Prédiction en lot (max 100) |
| `/examples` | GET | Exemples de transactions |
| `/docs` | GET | Documentation Swagger |

---

## 🔧 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/votre-username/fraud-detection.git
cd fraud-detection

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Entraîner les modèles (~10 min sur CPU)
jupyter notebook notebooks/fraud_detection_v2.ipynb

# 6. Lancer l'API
uvicorn src.api.main:app --reload --port 8000
```

### Tests

```bash
pytest tests/ -v
```

---

## 📁 Structure du projet

```
fraud-detection/
│
├── 📓 notebooks/
│   └── fraud_detection_v2.ipynb    # Pipeline complet ML (15 000 lignes)
│
├── 🐍 src/
│   └── api/
│       ├── main.py                 # FastAPI app
│       └── predictor.py            # Logique de prédiction (LSTM + XGBoost)
│
├── 🤖 models/                      # Modèles entraînés (générés)
│   ├── xgboost_model.pkl
│   ├── lstm_model.h5
│   ├── autoencoder.h5
│   ├── scaler.pkl
│   ├── ae_threshold.npy
│   ├── feature_names.json
│   └── ensemble_config.json        # {"w_lstm": 0.35, "w_xgb": 0.65}
│
├── 📊 docs/
│   ├── eda_plots.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── model_comparison.png
│
├── 🧪 tests/
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **ML/DL :** TensorFlow/Keras (LSTM + Autoencoder), XGBoost, Scikit-learn
- **Déséquilibre :** SMOTE (imbalanced-learn)
- **Explicabilité :** SHAP
- **API :** FastAPI + Pydantic + Uvicorn

---

## 📝 Licence

MIT — voir [LICENSE](LICENSE).

---

## 👤 Auteur

**Votre Nom**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/votre-profil)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/votre-username)

---

*⭐ Si ce projet vous a aidé, n'hésitez pas à mettre une étoile !*
