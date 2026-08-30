from collections.abc import Callable

import pytest

from finbehavior.models.feed_forward import FeedForward
from finbehavior.models.self_attention import SelfAttention
from finbehavior.models.self_attention_block import SelfAttentionBlock
from finbehavior.models.transformer_block import TransformerBlock

TransformerBlockFactory = Callable[[], TransformerBlock]


@pytest.fixture
def transformer_block_factory() -> TransformerBlockFactory:
    def build_transformer_block() -> TransformerBlock:
        self_attention = SelfAttention()

        self_attention_block = SelfAttentionBlock(
            self_attention=self_attention,
        )

        feed_forward = FeedForward()

        return TransformerBlock(
            self_attention_block=self_attention_block,
            feed_forward=feed_forward,
        )

    return build_transformer_block
