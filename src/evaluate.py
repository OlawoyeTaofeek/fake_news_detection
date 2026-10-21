"""
Phase 2: Evaluate the trained model on the held-out TEST set.

The test set is never touched during training or tuning, so these numbers are
the honest estimate of how the model will perform on new articles.

Usage:
    python -m src.evaluate
    python -m src.evaluate --min-f1 0.85     # exit with an error if F1 is too low (used in CI)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

REPORTS_DIR = Path("reports")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-path", default="models/model.joblib")
    parser.add_argument("--min-f1", type=float, default=0.0,
                        help="Fail (exit code 1) if test F1 is below this. Acts as a quality gate.")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    test = pd.read_csv(Path(args.data_dir) / "test.csv")
    preds = model.predict(test["content"])

    report = classification_report(
        test["label"], preds, target_names=["real", "fake"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(test["label"], preds, labels=[0, 1])
    f1 = f1_score(test["label"], preds, zero_division=0)

    print(classification_report(test["label"], preds, target_names=["real", "fake"], zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("              pred_real  pred_fake")
    print(f"actual_real   {cm[0][0]:9d}  {cm[0][1]:9d}")
    print(f"actual_fake   {cm[1][0]:9d}  {cm[1][1]:9d}")

    # Show a few mistakes: reading them is the best way to understand the model
    test = test.assign(pred=preds)
    mistakes = test[test["label"] != test["pred"]].head(5)
    if len(mistakes):
        print("\nSome mistakes to study:")
        for _, row in mistakes.iterrows():
            actual = "fake" if row["label"] == 1 else "real"
            print(f"  [actual={actual}] {row['content'][:120]}...")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = {
        "test_rows": len(test),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved metrics to {REPORTS_DIR}/test_metrics.json")

    if f1 < args.min_f1:
        print(f"FAILED quality gate: F1 {f1:.4f} < required {args.min_f1}")
        sys.exit(1)
    print(f"Test F1: {f1:.4f}")


if __name__ == "__main__":
    main()
