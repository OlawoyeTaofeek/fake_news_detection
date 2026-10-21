"""
Prediction logging. Every prediction is saved to SQLite so we can monitor the model later.

SQLite is a single file, needs no server, and is perfect for learning.
In production you would swap this for Postgres (the SQL stays almost the same).
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id    TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    text          TEXT NOT NULL,
    word_count    INTEGER NOT NULL,
    oov_ratio     REAL NOT NULL,
    label         TEXT NOT NULL,
    prob_fake     REAL NOT NULL,
    confidence    REAL NOT NULL,
    latency_ms    REAL NOT NULL,
    model_version TEXT NOT NULL,
    feedback      INTEGER            -- NULL = no feedback, 1 = correct, 0 = wrong
);
"""


def db_path() -> Path:
    return Path(os.getenv("DB_PATH", "logs/predictions.db"))


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def log_prediction(**row) -> None:
    row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    cols = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    with connect() as conn:
        conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({placeholders})", tuple(row.values()))


def save_feedback(request_id: str, correct: bool) -> bool:
    """Returns False if the request_id does not exist."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE predictions SET feedback = ? WHERE request_id = ?",
            (1 if correct else 0, request_id),
        )
        return cur.rowcount > 0


def get_stats() -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)                                     AS total,
                   AVG(CASE WHEN label = 'fake' THEN 1.0 ELSE 0.0 END) AS fake_ratio,
                   AVG(confidence)                              AS avg_conf,
                   AVG(latency_ms)                              AS avg_latency,
                   COUNT(feedback)                              AS fb_count,
                   AVG(feedback)                                AS fb_acc
            FROM predictions
            """
        ).fetchone()

    def r(x):
        return None if x is None else round(float(x), 4)

    return {
        "total_predictions": row["total"],
        "fake_ratio": r(row["fake_ratio"]),
        "avg_confidence": r(row["avg_conf"]),
        "avg_latency_ms": r(row["avg_latency"]),
        "feedback_count": row["fb_count"],
        "feedback_accuracy": r(row["fb_acc"]),
    }
