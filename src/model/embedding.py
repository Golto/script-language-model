import math
import torch
import torch.nn as nn
from .config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal fixe (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                     # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embedding(positions))


class TokenEmbedding(nn.Module):
    """
    Lookup table + encodage positionnel.
    Sortie : (batch, seq_len, d_model)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.pos_encoding = LearnedPositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )
        # Mise à l'échelle standard (Vaswani)
        self.scale = math.sqrt(config.d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids : (batch, seq_len)
        x = self.embedding(token_ids) * self.scale
        return self.pos_encoding(x)