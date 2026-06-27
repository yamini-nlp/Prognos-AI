# 🩺 PrognosAI — Clinical NLP Inference Platform

> An end-to-end clinical NLP system that analyses free-text hospital discharge summaries and predicts clinical outcomes using three model architectures: a TF-IDF baseline, a text-plus-vitals hybrid model, and a Groq LLaMA-3 70B large language model.

**Live Demo:** https://prognos-ai-five.vercel.app &nbsp;|&nbsp; **GitHub:** github.com/yamini-nlp/Prognos-AI

![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Next.js%20%7C%20scikit--learn-blue?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B%20%7C%20Groq-orange?style=flat-square)
![Models](https://img.shields.io/badge/Models-Baseline%20%7C%20Hybrid%20%7C%20LLM-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## Overview

PrognosAI demonstrates how unstructured clinical text can be transformed into actionable predictions using a unified ML and LLM-driven pipeline. Three independent inference pipelines operate on the same clinical note input, allowing direct comparison of classical ML, feature-engineered hybrid, and LLM-based approaches on identical inputs. An explainability-first design surfaces phrase-level reasoning for every prediction, making outputs transparent and auditable.

> ⚠️ **Important:** Models in this system are trained on synthetically generated discharge notes. Performance figures are not representative of real-world clinical data. The system is a research and engineering prototype and is **not** suitable for clinical use.

---

## 🎯 Problem Statement

Hospital discharge summaries contain rich clinical information that is rarely utilised for predictive analytics at scale. Manually reviewing notes for triage decisions, readmission screening, or specialty routing is time-consuming and inconsistent across clinical teams.

This project explores three approaches to:
- Predicting 30-day readmission risk from free-text discharge notes
- Classifying length of stay into short (<3 days), medium (3–7 days), and long (>7 days) bands
- Identifying the medical specialty from unstructured clinical text
- Returning explainable outputs — highlighted phrases, confidence scores, and clinical risk factors — for every prediction

---

## 📋 Dataset

| Property | Detail |
|---|---|
| Source | Synthetic generator (`backend/data/synthetic_generator.py`) |
| Size | 1,200 synthetic discharge notes across 5 specialties |
| Structured features | Age, gender, vitals, comorbidity count, prior admissions |
| Label assignment | Probabilistic based on clinical templates, not deterministic |

**Prediction tasks:**

| Task | Type | Classes |
|---|---|---|
| Readmission Risk | Binary | No Readmission / 30-Day Readmission |
| Length of Stay | 3-class | Short (<3d) / Medium (3–7d) / Long (>7d) |
| Medical Specialty | 5-class | Cardiology / Neurology / Orthopedics / Oncology / General Medicine |

---

## 🏗️ Three Inference Pipelines

```
User Input (clinical note + optional vitals)
        │
        ├─► [Baseline Pipeline]
        │   Preprocessing → TF-IDF (20k features, trigrams, sublinear TF)
        │   → Logistic Regression → Prediction + token importance scores
        │
        ├─► [Hybrid Pipeline]
        │   Preprocessing → TF-IDF (15k features)
        │   + 18 engineered vitals features (sparse hstack)
        │   → Logistic Regression → Prediction
        │
        └─► [Groq LLM Pipeline]
            Structured system prompt → LLaMA-3.3 70B
            → JSON response → Prediction + Reasoning + Risk Factors
                    │
                    ▼
        FastAPI /predict → Explainability payload → Next.js frontend
```

---

## ⚙️ Model Architecture

**Baseline model:**

| Component | Detail |
|---|---|
| Vectoriser | TF-IDF, 20,000 features, 1–3 ngrams, sublinear TF scaling |
| Classifier | Logistic Regression, C=1.0, balanced class weights |
| Explainability | Per-token TF-IDF × logistic regression coefficient score |

**Hybrid model:**

| Component | Detail |
|---|---|
| Text features | TF-IDF, 15,000 features, 1–2 ngrams |
| Tabular features | 18 engineered vitals and demographic features |
| Fusion | `scipy.sparse.hstack` (text matrix + scaled tabular) |
| Classifier | Logistic Regression, C=0.8, balanced class weights |

**Vitals feature engineering (hybrid model):** raw vitals augmented into 18 derived features including pulse pressure, shock index, elderly age flag, tachycardia, hypotension, hypoxia, and tachypnoea indicators — all scaled with `StandardScaler` before sparse fusion.

**Groq LLM:**

| Component | Detail |
|---|---|
| Model ID | `llama-3.3-70b-versatile` via Groq API |
| Prompting | Task-specific system prompts with JSON-only response enforcement |
| Primary output | Prediction class + confidence float + clinical reasoning string |
| Explanation call | Secondary API call extracts key phrase importance scores |
| Risk output | Structured risk factor and protective factor lists |

---

## 📊 Results

| Task | Baseline Accuracy | Hybrid Accuracy | Notes |
|---|---|---|---|
| Readmission Risk | ~54% | ~60% | Probabilistic labels; reflects task difficulty on synthetic data |
| Length of Stay | ~41% | ~37% | Three-class task; probabilistic labels |
| Medical Specialty | ~100% | ~100% | Trivially explained by distinct vocabulary per specialty in synthetic data; not a meaningful benchmark |

**Groq LLM outputs** are consistently clinically coherent in structure and correctly identify specialty and readmission signals in synthetic notes. Quantitative accuracy on the LLM pipeline is not computed — outputs are qualitative and structured for human review.

*All results are from models trained on synthetic data. Performance on real clinical datasets (e.g. MIMIC-III) would differ and requires formal evaluation.*

---

## 🔍 Explainability

All three pipelines return phrase-level explanations rendered in the UI:

- **Baseline / Hybrid:** TF-IDF × coefficient scores highlight tokens by importance tier (high / medium / low) with colour-coded annotation
- **Groq LLM:** a secondary inference call extracts clinically significant phrases with natural language explanations
- **ResultsPanel:** confidence ring visualisation, probability bar chart, annotated note, key phrase chips, and structured risk/protective factor cards

---

## ⚠️ Limitations

- **Synthetic training data:** models trained on generated notes; performance on real MIMIC or hospital EHR data will differ significantly and has not been evaluated
- **Baseline model accuracy:** TF-IDF + Logistic Regression is not competitive with transformer-based models on complex clinical NLP tasks
- **Specialty classification result:** ~100% accuracy is a consequence of distinct synthetic vocabulary per specialty, not a meaningful benchmark result
- **Groq rate limits:** free API tier has per-minute request limits that may slow responses under concurrent load
- **No authentication:** the API has no auth layer; not suitable for production clinical use without security hardening
- **Ephemeral model storage:** on Render's free tier, trained `.pkl` files are lost on redeploy; persistent disk storage is required for production

---

## 🔭 Future Work

- Train on MIMIC-III or real de-identified clinical notes for meaningful benchmarking
- Fine-tune ClinicalBERT for improved accuracy on clinical NLP tasks
- Add user authentication and session management for production deployment
- Implement SHAP values for rigorous ML explainability alongside token highlights
- Containerise with Docker Compose for consistent local and cloud deployment
- Add batch prediction endpoint for processing multiple notes simultaneously

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| ML — Text | scikit-learn TF-IDF + Logistic Regression |
| ML — Hybrid | TF-IDF + engineered vitals features (scipy sparse hstack) |
| LLM | Groq API — LLaMA-3.3 70B Versatile |
| NLP Preprocessing | NLTK (tokenisation, stopwords, lemmatisation) |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Charts | Recharts |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |

---

## 🚀 Local Setup

**Prerequisites:** Python 3.11+ · Node.js 20+ · Groq API key (free at [console.groq.com](https://console.groq.com))

**1. Clone**
```bash
git clone https://github.com/yamini-nlp/Prognos-AI.git
cd Prognos-AI
```

**2. Backend setup**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:
```
GROQ_API_KEY=gsk_your_key_here
MODEL_DIR=./saved_models
DATA_DIR=./data
ENV=development
```

**3. Train models**
```bash
python3 -m training.train
```
Accuracy scores will print for all three tasks. Takes approximately 60 seconds.

**4. Start backend**
```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Verify at `http://localhost:8000` — should return `{"status":"ok"}`

**5. Frontend setup**
```bash
cd ../frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```
Visit `http://localhost:3000`

---

## 📁 Repository Structure

```
Prognos-AI/
├── backend/
│   ├── data/
│   │   └── synthetic_generator.py      # Generates 1,200 synthetic clinical notes
│   ├── models/
│   │   ├── baseline_model.py           # TF-IDF + Logistic Regression
│   │   ├── groq_model.py               # LLaMA-3.3 70B via Groq API
│   │   └── hybrid_model.py             # TF-IDF + tabular feature fusion
│   ├── training/
│   │   └── train.py                    # Training pipeline CLI
│   ├── utils/
│   │   ├── text_processing.py          # Tokenisation, lemmatisation
│   │   ├── feature_engineering.py      # Vitals feature engineering (18 features)
│   │   ├── explainability.py           # Token highlighting, phrase extraction
│   │   └── metrics.py                  # Accuracy, F1, confusion matrix
│   ├── saved_models/                   # .pkl files — generated after training
│   ├── main.py                         # FastAPI — /predict /explain /train
│   └── requirements.txt
├── frontend/
│   ├── components/
│   │   ├── charts/ProbabilityChart.tsx
│   │   ├── ui/ResultsPanel.tsx
│   │   ├── HighlightedNote.tsx
│   │   └── TabularInputs.tsx
│   ├── pages/
│   │   ├── index.tsx                   # Landing page
│   │   ├── predict.tsx                 # Prediction console
│   │   └── train.tsx                   # Training dashboard
│   └── package.json
└── README.md
```

---

*Built by Yamini G &nbsp;·&nbsp; [GitHub](https://github.com/yamireddy04/Prognos-AI) &nbsp;·&nbsp; [Live Demo](https://prognos-ai-five.vercel.app)*
