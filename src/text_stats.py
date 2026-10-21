"""Small text statistics shared by the API (when logging) and the drift report (when comparing)."""

from sklearn.pipeline import Pipeline


def word_count(text: str) -> int:
    return len(text.split())


def oov_ratio(model: Pipeline, cleaned_text: str) -> float:
    """Share of the text's terms that the model has never seen in training.

    A rising out-of-vocabulary (OOV) ratio over time means users are sending
    language the model was not trained on: a classic sign of data drift.
    """
    tfidf = model.named_steps["tfidf"]
    analyzer = tfidf.build_analyzer()
    terms = [t for t in analyzer(cleaned_text) if " " not in t]  # single words only
    if not terms:
        return 1.0
    vocab = tfidf.vocabulary_
    unknown = sum(1 for t in terms if t not in vocab)
    return round(unknown / len(terms), 4)
