import pytest
import torch
from phase1_embedding_foundation import EmbeddingLayer, RoundingLayer


VOCAB_SIZE = 1000
EMBED_DIM = 64
BATCH, SEQ_LEN = 4, 16


@pytest.fixture
def embed_layer():
    return EmbeddingLayer(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)


@pytest.fixture
def rounding_layer(embed_layer):
    return RoundingLayer(embedding_weight=embed_layer.weight)


def test_embedding_output_shape(embed_layer):
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    out = embed_layer(ids)
    assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)


def test_embedding_loss_is_scalar(embed_layer):
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    x0_pred = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    loss = embed_layer.embedding_loss(x0_pred, ids)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_soft_rounding_shape(rounding_layer):
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    logits = rounding_layer.soft_round(x)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)


def test_hard_rounding_ids_in_range(rounding_layer):
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    ids = rounding_layer.hard_round(x)
    assert ids.shape == (BATCH, SEQ_LEN)
    assert ids.min() >= 0
    assert ids.max() < VOCAB_SIZE
