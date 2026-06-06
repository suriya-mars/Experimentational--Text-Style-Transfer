import pytest
from phase7_evaluation import ContentPreservation, Fluency, SelfBLEU


SOURCES = ["The food was absolutely terrible and cold.", "This place is fantastic and friendly."]
OUTPUTS = ["The food was great and warm.", "This place is awful and rude."]


def test_content_preservation_returns_float():
    metric = ContentPreservation()
    score = metric(SOURCES, OUTPUTS)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_self_bleu_lower_for_diverse():
    diverse = ["The cat sat on the mat.", "Rain falls on the plains of Spain.", "A bird flew over the mountain."]
    repetitive = ["The cat sat.", "The cat sat here.", "The cat sat there."]
    metric = SelfBLEU()
    assert metric(diverse) < metric(repetitive)


def test_self_bleu_single_text():
    metric = SelfBLEU()
    score = metric(["Only one sentence here."])
    assert score == 0.0
