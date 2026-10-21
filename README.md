# 🔎 Fake News Detector: an end-to-end ML project

A complete machine learning system that flags **misinformation-style writing**, built the way
real ML products are built: data pipeline → training with experiment tracking → REST API →
web frontend → Docker → CI/CD → deployment → monitoring for data drift.

> **Honest scope:** the model learns *writing patterns* common in fake news (sensational
> language, emotional appeals, lack of sourcing). It does **not** check facts. A false claim
> written calmly can pass; a true story written sensationally can be flagged.

**Live demo:** _add your link here_ · **Demo video:** _add your link here_

---

## Architecture

```
┌────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Streamlit │ ──▶ │   FastAPI    │ ──▶ │ TF-IDF + LogReg   │
│  frontend  │ ◀── │   backend    │ ◀── │ (model.joblib)    │
└────────────┘     └──────┬───────┘     └───────────────────┘
                          │ logs every prediction + user feedback
                          ▼
                   ┌──────────────┐     ┌───────────────────┐
                   │   SQLite     │ ──▶ │  Drift report     │ ──▶ retrain?
                   │ prediction   │     │  (5 checks, HTML) │
                   │ log          │     └───────────────────┘
                   └──────────────┘
```

| Layer | Tools |
|---|---|
| Data & training | pandas, scikit-learn, MLflow (optional) |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Monitoring | SQLite prediction log, KS-tests (SciPy), HTML drift report |
| Packaging | Docker, Docker Compose |
| CI/CD | GitHub Actions, Render |
| Testing | pytest |

---

## Quick start (5 minutes, no download needed)

Requires **Python 3.11–3.13**.

```bash
# 1. Set up
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Prepare the small sample dataset, train and evaluate
python -m src.preprocess --raw-dir data/sample
python -m src.train
python -m src.evaluate

# 3. Run the tests
pytest -v

# 4. Start the API (terminal 1)
uvicorn api.main:app --reload
#    -> open http://127.0.0.1:8000/docs

# 5. Start the frontend (terminal 2)
streamlit run frontend/app.py
#    -> open http://localhost:8501
```

The sample data is synthetic and very easy, so you'll see ~100% accuracy. That's expected.
Once everything works, switch to the real dataset below.

---

## Using the real dataset

1. Download the **ISOT Fake News dataset** from Kaggle (`Fake.csv` and `True.csv`).
2. Put both files in `data/raw/`.
3. Run:

```bash
python -m src.preprocess
python -m src.train
python -m src.evaluate --min-f1 0.85
```

See [`data/README.md`](data/README.md) for details, including the **Reuters data-leakage trap**.

---

## Project walkthrough

### Phase 1: Data (`src/preprocess.py`)
- Combines title + body, lowercases, removes URLs and Twitter handles.
- **Removes the `CITY (Reuters) -` dateline** that otherwise leaks the label.
- Drops empty rows and duplicates, then makes a stratified 70/15/15 train/val/test split.
- `clean_text()` is reused by the API, so training and serving prepare text identically.

### Phase 2: Training & evaluation (`src/train.py`, `src/evaluate.py`)
- TF-IDF (single words + word pairs) → logistic regression with balanced class weights.
- Saves `models/model.joblib` plus `models/metadata.json` (version, metrics, top fake/real words).
- Logs parameters and metrics to **MLflow** if it's installed (`pip install mlflow`, then `mlflow ui`).
- `evaluate.py` reports precision/recall/F1, a confusion matrix, prints example mistakes,
  and can act as a **quality gate** (`--min-f1`) that fails CI if the model is too weak.

Try experiments: `python -m src.train --C 0.5`, `--max-features 20000`, `--ngram-max 1`.

### Phase 3: Backend (`api/`)

| Endpoint | Purpose |
|---|---|
| `POST /predict` | `{"text": "..."}` → label, confidence, probability, request id |
| `POST /feedback` | `{"request_id": "...", "correct": true}`: the user says if the prediction was right |
| `GET /health` | Is the service up and is the model loaded? |
| `GET /stats` | Live numbers: predictions served, % flagged fake, avg confidence, latency, feedback accuracy |

Input is validated by Pydantic (20–20,000 characters). Every prediction is logged to SQLite.

### Phase 4: Frontend (`frontend/app.py`)
Text box with examples, colour-coded result card, probability bar, disclaimer,
👍/👎 feedback buttons, and a sidebar showing live stats from the API.

### Phase 5: Docker & CI/CD
```bash
python -m src.train                 # a model must exist before building
docker compose up --build
# Frontend: http://localhost:8501   API: http://localhost:8000/docs
```
Prediction logs are saved to `./logs` on your machine.

`.github/workflows/ci.yml` runs on every push: train on sample data → quality gate →
tests → build both Docker images → smoke-test the API container → deploy (on `main`).

### Phase 6: Monitoring (`monitoring/`)
```bash
python -m monitoring.simulate_traffic --n 100             # normal traffic
python -m monitoring.drift_report                         # ✅ no drift
python -m monitoring.simulate_traffic --n 100 --mode drift  # off-topic traffic
python -m monitoring.drift_report                         # ⚠️ drift detected
```
Opens as `reports/drift_report.html`. It compares recent production traffic against the
validation set on five signals:

| Check | Method | What drift means |
|---|---|---|
| Input length | KS test | People send much longer/shorter text than we trained on |
| Unknown-word ratio | KS test | New topics, slang or languages the model never saw |
| Prediction confidence | KS test | The model is less sure, usually because input is unfamiliar |
| Share predicted fake | Difference > 20 pts | The output balance shifted |
| Feedback accuracy | Drop > 10 pts | Users report more wrong predictions |

`--fail-on-drift` exits with code 2, so a scheduled job can trigger retraining.

---

## Deploying to Render (free)

1. Train on the real data and **commit `models/model.joblib`** (a few MB).
2. Push the repo to GitHub.
3. In Render: **New → Blueprint** → select the repo. `render.yaml` creates both services.
4. When the API is live, set the frontend's `API_URL` to the API's URL
   (e.g. `https://fake-news-api-xxxx.onrender.com`).
5. Optional auto-deploy from CI: copy each service's **Deploy Hook** URL into GitHub secrets
   `RENDER_API_DEPLOY_HOOK` and `RENDER_FRONTEND_DEPLOY_HOOK`.

> Free Render services sleep when idle (first request is slow) and their disk is wiped on
> restart, so the SQLite log resets. For persistent monitoring, use a Render disk or Postgres.

Alternatives: Hugging Face Spaces (Docker Space), Railway, Google Cloud Run.

---

## Project structure

```
fake-news-detector/
├── data/
│   ├── README.md             # how to get the dataset, leakage notes, DVC
│   └── sample/               # small synthetic dataset for tests and CI
├── src/
│   ├── preprocess.py         # cleaning, leakage removal, splitting
│   ├── train.py              # training + MLflow tracking
│   ├── evaluate.py           # test-set metrics + quality gate
│   └── text_stats.py         # shared helpers for monitoring
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # request/response validation
│   └── database.py           # prediction logging (SQLite)
├── frontend/
│   └── app.py                # Streamlit app
├── monitoring/
│   ├── drift_report.py       # drift checks + HTML report
│   └── simulate_traffic.py   # generates traffic for demos
├── tests/                    # pytest: preprocessing, model, API
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── render.yaml               # Render deployment blueprint
├── Makefile                  # shortcuts: make train, make test, make docker...
├── .github/workflows/ci.yml
└── README.md
```

---

## Ideas to extend it

- **Better model:** fine-tune DistilBERT and compare it with the baseline in MLflow.
- **Explainability:** highlight which words pushed the prediction towards "fake".
- **Automatic retraining:** a scheduled GitHub Action that runs the drift report and retrains
  using feedback data when drift is found.
- **Grafana dashboard:** expose metrics with `prometheus-fastapi-instrumentator`.
- **Postgres** instead of SQLite for persistent logs in production.
- **React/Next.js frontend** calling the same API.

## Limitations

- Learns style, not truth: it cannot verify claims.
- Trained mostly on 2016–2017 US political news; performance drops on other countries,
  topics and time periods. That is exactly what the monitoring is designed to catch.
- Prediction logs store submitted text. In a real product, tell users and set a retention policy.

## License

MIT
