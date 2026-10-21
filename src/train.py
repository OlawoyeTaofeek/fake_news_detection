"""
Phase 2: Train the model and track the experiment.

Pipeline: TF-IDF (turns text into numbers) -> Logistic Regression (learns fake vs real).
Every run is logged to MLflow (if installed) so you can compare experiments.

Usage:
    python -m src.train
    python -m src.train --max-features 20000 --C 2.0      # try different settings
    mlflow ui                                             # then open http://127.0.0.1:5000
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:  # MLflow is optional: training still works without it
    MLFLOW_AVAILABLE = False

MODEL_DIR = Path("models")

def build_pipeline(max_features: int, C: float, ngram_max: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),  # single words and word pairs
                    min_df=2,                    # ignore words that appear in only 1 article
                    max_df=0.9,                  # ignore words that appear almost everywhere
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(C=C, max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

def score(model: Pipeline, X, y) -> dict:
    preds = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall": round(recall_score(y, preds, zero_division=0), 4),
        "f1": round(f1_score(y, preds, zero_division=0), 4),
    }

def top_words(model: Pipeline, n: int = 15) -> dict:
    """The words that push hardest towards 'fake' and towards 'real'. Great for the README."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = coefs.argsort()
    return {
        "fake": [str(vocab[i]) for i in order[::-1][:n]],
        "real": [str(vocab[i]) for i in order[:n]],
    }

def main():
    parser = argparse.ArgumentParser(description="Train the fake news model")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--experiment", default="fake-news-detector")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    print(f"Training on {len(train)} rows, validating on {len(val)} rows")

    params = {"max_features": args.max_features, "C": args.C, "ngram_max": args.ngram_max}
    model = build_pipeline(**params)

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run()

    model.fit(train["content"], train["label"])
    train_metrics = score(model, train["content"], train["label"])
    val_metrics = score(model, val["content"], val["label"])

    print("Train metrics:     ", train_metrics)
    print("Validation metrics:", val_metrics)
    if train_metrics["f1"] - val_metrics["f1"] > 0.05:
        print("WARNING: big gap between train and validation F1, the model may be overfitting.")

    # --- Save the model + metadata the API and monitoring will need -----------
    MODEL_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    joblib.dump(model, MODEL_DIR / "model.joblib")

    lengths = train["content"].str.split().str.len()
    metadata = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "train_rows": len(train),
        "val_metrics": val_metrics,
        "top_words": top_words(model),
        # Reference statistics used later by monitoring/drift_report.py
        "reference": {
            "fake_ratio": round(float(train["label"].mean()), 4),
            "word_count_mean": round(float(lengths.mean()), 2),
            "word_count_median": float(lengths.median()),
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model version {version} to {MODEL_DIR}/model.joblib")

    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"))
        try:
            mlflow.sklearn.log_model(model, name="model")           # MLflow 3.x
        except TypeError:
            mlflow.sklearn.log_model(model, artifact_path="model")  # MLflow 2.x
        mlflow.end_run()
        print(f"Logged run {run.info.run_id} to MLflow. Run `mlflow ui` to view it.")
    else:
        print("(MLflow not installed, skipping experiment tracking. `pip install mlflow` to enable.)")


if __name__ == "__main__":
    main()
