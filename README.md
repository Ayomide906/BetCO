# ⚽ BetCO: Live Match Prediction Engine 🤖

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Powered-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

Welcome to the brain behind **BetCO**! This repository houses a production-ready FastAPI machine learning engine designed to predict Premier League match outcomes, calculate exact goal probabilities using Poisson distribution, and explain its reasoning using SHAP AI explainers.

---

## 🧭 Interactive Navigation
*Click on any section below to expand and explore the architecture!*

<details>
<summary><b>✨ Core Features</b></summary>
<br>

* **🧠 CatBoost Machine Learning:** Dual-layer models predicting both W/D/L outcomes and Expected Goals (xG).
* **🎯 Poisson Goal Markets:** Translates raw expected goals into real mathematical probabilities for Over/Under markets (Total, Home, and Away).
* **🕵️‍♂️ SHAP AI Explainability:** Transparent predictions that tell users *exactly* which stats drove the model's confidence.
* **⚡ Live Odds Integration:** Asynchronously fetches real-time market odds (e.g., Bet365) to ensure predictions adapt to live market movements.
* **🚀 O(1) Feature Engineering:** Pre-indexes historical match datasets in memory on server startup for lightning-fast live inference.

</details>

<details>
<summary><b>📂 Repository Architecture</b></summary>
<br>

A clean separation of concerns for scalable MLOps:

```text
BetCO/
│
├── app/                      # 🌐 Web/API Layer
│   ├── __init__.py
│   └── app.py                # FastAPI endpoints and async handlers
│
├── src/                      # 🧠 Core ML Logic & Data Pipelines
│   ├── __init__.py
│   ├── feature_engineer.py   # LiveMatchFeatureEngineer class
│   ├── train.py              # Automated CT/CD training scripts
│   └── update_features.py    # Data ingestion scripts
│
├── model/                    # 📦 Artifact Vault (Ignored by Git)
│   ├── result.pkl            # Baseline W/D/L Model
│   ├── Homegoals_model.pkl              # Baseline Home Goals Model
│   └── Awaygoals_model.pkl              # Baseline Away Goals Model
│
└── data/                     # 🗄️ Datasets (Ignored by Git)
    ├── df1_raw.csv           # Match history
    └── team_df_raw.csv       # Overall team form tracking