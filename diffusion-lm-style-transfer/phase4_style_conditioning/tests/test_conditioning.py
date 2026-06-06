import pytest
import torch
from phase4_style_conditioning import StyleClassifier, ClassifierGuidance


BATCH, SEQ, DIM, NUM_STYLES = 4, 16, 64, 2


@pytest.fixture
def classifier():
    return StyleClassifier(embed_dim=DIM, num_styles=NUM_STYLES, hidden_dim=128)


@pytest.fixture
def guidance(classifier):
    return ClassifierGuidance(classifier, guidance_scale=2.0)


def test_classifier_output_shape(classifier):
    x = torch.randn(BATCH, SEQ, DIM)
    logits = classifier(x)
    assert logits.shape == (BATCH, NUM_STYLES)


def test_classifier_log_prob_shape(classifier):
    x = torch.randn(BATCH, SEQ, DIM)
    labels = torch.randint(0, NUM_STYLES, (BATCH,))
    log_p = classifier.log_prob(x, labels)
    assert log_p.shape == (BATCH,)


def test_guidance_shifts_embedding(guidance):
    x0_hat = torch.randn(BATCH, SEQ, DIM)
    target = torch.zeros(BATCH, dtype=torch.long)
    guided = guidance.guide(x0_hat, target)
    assert guided.shape == x0_hat.shape
    assert not torch.allclose(guided, x0_hat)
