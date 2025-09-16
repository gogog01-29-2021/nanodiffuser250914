"""A stub text encoder for tiny diffusion models.

This minimal text encoder provides a placeholder implementation so that
pipelines expecting a text encoder can run without a full language model.
When input IDs are provided, it returns learned embeddings; if no input is
given, it returns zero vectors of the appropriate shape.

Args:
    vocab_size (int): number of tokens in the fake vocabulary.
    emb_dim (int): dimension of the embedding vectors.
    max_length (int): maximum sequence length; used when returning zeros.
"""

import torch
import torch.nn as nn


class TextEncoderStub(nn.Module):
    """A very simple embedding lookup layer used as a text encoder stub."""

    def __init__(self, vocab_size: int = 1, emb_dim: int = 768, max_length: int = 77):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Return embeddings for the provided token IDs or zeros if None/empty.

        Args:
            input_ids (Tensor or None): token IDs of shape [batch, seq_len] or None.

        Returns:
            Tensor: embeddings of shape [batch, seq_len, emb_dim] or zeros of
            shape [batch, max_length, emb_dim] if no input IDs are provided.
        """
        if input_ids is None or input_ids.numel() == 0:
            batch_size = 1 if input_ids is None else input_ids.shape[0]
            return torch.zeros(
                batch_size, self.max_length, self.embedding.embedding_dim,
                device=self.embedding.weight.device if input_ids is None else input_ids.device
            )
        return self.embedding(input_ids)
