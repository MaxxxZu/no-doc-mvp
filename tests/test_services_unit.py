def test_classify_service_unit():
    from config import Settings
    from services.classification_service import classify_text_raw

    def fake(**kwargs):
        return {"labels": kwargs["candidate_labels"], "scores": [0.8, 0.1, 0.1]}

    labels, scores = classify_text_raw(
        "Dette er en test", fake, Settings(_env_file=None), ["A", "B", "C"], False,
    )
    assert labels == ["A", "B", "C"]
    assert scores == [0.8, 0.1, 0.1]


def test_ner_service_unit():
    from services.ner_service import extract_entities_raw

    expected = [{"entity_group": "PER", "word": "Ola Nordmann", "score": 0.95,
                 "start": 0, "end": 12}]
    assert extract_entities_raw("Ola Nordmann", lambda text: expected) == expected
