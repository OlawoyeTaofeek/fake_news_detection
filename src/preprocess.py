"""
Phase 1: Data preparation.

Loads the ISOT Fake News dataset (Fake.csv + True.csv), cleans the text,
removes data leakage, de-duplicates, and splits into train / val / test.

Usage:
    python -m src.preprocess                         # uses data/raw/
    python -m src.preprocess --raw-dir data/sample   # uses the tiny sample (for CI / quick tests)
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42

# Label convention used everywhere in this project
FAKE = 1
REAL = 0

# --- Leakage patterns -------------------------------------------------------
# Almost every REAL article in ISOT starts with "CITY (Reuters) - ".
# If we leave it in, the model learns "contains Reuters = real" and gets ~99%
# accuracy while learning nothing useful about misinformation.
# Matches e.g. "WASHINGTON (Reuters) - " or "LONDON/PARIS (Reuters) - " at the start of an article body.
REUTERS_PREFIX = re.compile(r"^\s*[A-Z][A-Za-z .,/'-]{0,60}\(Reuters\)\s*[-–—]\s*")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
TWITTER_HANDLE = re.compile(r"@\w+")
PIC_LINK = re.compile(r"pic\.twitter\.com/\S+")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean one piece of text.

    IMPORTANT: the API calls this exact same function before predicting,
    so training and serving always see text prepared the same way.
    """
    if not isinstance(text, str):
        return ""
    text = REUTERS_PREFIX.sub("", text)
    text = URL_PATTERN.sub(" ", text)
    text = PIC_LINK.sub(" ", text)
    text = TWITTER_HANDLE.sub(" ", text)
    text = text.replace("(Reuters)", " ").replace("Reuters", " ")
    text = text.lower()
    text = WHITESPACE.sub(" ", text).strip()
    return text


def load_raw(raw_dir: Path) -> pd.DataFrame:
    fake_path, true_path = raw_dir / "Fake.csv", raw_dir / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Expected Fake.csv and True.csv in '{raw_dir}'. "
            "Download the ISOT dataset (see data/README.md) or run with --raw-dir data/sample"
        )
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    fake["label"] = FAKE
    real["label"] = REAL
    return pd.concat([fake, real], ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("")
    # Remove the "CITY (Reuters) - " dateline from the body BEFORE joining with the title
    df["text"] = df["text"].fillna("").astype(str).map(lambda t: REUTERS_PREFIX.sub("", t))
    # Title + body together gives the model more signal
    df["content"] = (df["title"] + ". " + df["text"]).map(clean_text)

    before = len(df)
    df = df[df["content"].str.len() >= 20]          # drop empty / near-empty rows
    df = df.drop_duplicates(subset="content")        # ISOT has many duplicates
    print(f"Rows: {before} -> {len(df)} after removing empty rows and duplicates")

    return df[["content", "label"]].reset_index(drop=True)


def split(df: pd.DataFrame):
    # 70% train, 15% validation, 15% test, keeping the fake/real ratio the same in each
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=RANDOM_STATE
    )
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare the fake news dataset")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder with Fake.csv and True.csv")
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare(load_raw(Path(args.raw_dir)))
    train, val, test = split(df)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out_dir / f"{name}.csv", index=False)
        fake_share = part["label"].mean() * 100
        print(f"  {name:5s}: {len(part):6d} rows  ({fake_share:.1f}% fake)")

    print(f"Saved processed data to {out_dir}/")


if __name__ == "__main__":
    main()
