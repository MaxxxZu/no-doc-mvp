"""Validation shared by inference services."""
import math
from numbers import Integral, Real


class InputTooLong(Exception):
    pass


def token_limit(pipeline):
    config = pipeline.model.config
    maximum = getattr(config, "max_position_embeddings", None)
    tokenizer_maximum = getattr(pipeline.tokenizer, "model_max_length", None)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        raise RuntimeError("Model position capacity is unavailable")
    if getattr(config, "model_type", None) not in {"bert", "roberta", "xlm-roberta"}:
        raise RuntimeError("Unsupported model position capacity")
    # RoBERTa-style position IDs start after the padding index.
    if getattr(config, "model_type", None) in {"roberta", "xlm-roberta"}:
        padding = getattr(config, "pad_token_id", None)
        if not isinstance(padding, int) or padding < 0:
            raise RuntimeError("Model position capacity is unavailable")
        maximum -= padding + 1
    # Transformers reserves values at or above 10**20 as effectively unbounded.
    if isinstance(tokenizer_maximum, int) and 0 < tokenizer_maximum < 10**20:
        maximum = min(maximum, tokenizer_maximum)
    if maximum <= 0:
        raise RuntimeError("Model position capacity is unavailable")
    return maximum


def check_tokens(pipeline, text, hypothesis=None):
    encoded = pipeline.tokenizer(text, hypothesis, add_special_tokens=True, truncation=False)
    if len(encoded["input_ids"]) > token_limit(pipeline):
        raise InputTooLong()


def valid_score(value):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def validate_classification(result, categories):
    labels, scores = result
    if (not isinstance(labels, (list, tuple)) or not isinstance(scores, (list, tuple))
            or not labels or len(labels) != len(scores)
            or any(not isinstance(label, str) for label in labels)
            or len(set(labels)) != len(labels) or set(labels) != set(categories)
            or any(not valid_score(score) for score in scores)):
        raise ValueError("Invalid classification result")
    return labels, scores


def validate_entities(predictions, text):
    if not isinstance(predictions, list):
        raise ValueError("Invalid NER result")
    entities = []
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Invalid NER result")
        group, score = item.get("entity_group"), item.get("score")
        start, end = item.get("start"), item.get("end")
        if (not isinstance(group, str) or not group
                or not isinstance(item.get("word"), str) or not valid_score(score)
                or not isinstance(start, Integral) or isinstance(start, bool)
                or not isinstance(end, Integral) or isinstance(end, bool)
                or not 0 <= start < end <= len(text)):
            raise ValueError("Invalid NER result")
        entities.append(dict(entity_group=group, word=text[start:end], score=score,
                             start=start, end=end))
    return entities
