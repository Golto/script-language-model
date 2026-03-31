import torch
from torch.utils.data import Dataset
from typing import List

from src.tokenizer import LanguageTokenizer
from src.data.snippet import CodeSnippet

from .config import ModelConfig


class ProgramDataset(Dataset):
    """
    Dataset de séquences pour la prédiction du prochain token.

    Chaque snippet est encodé en [BOS, t1, ..., tn, EOS].
    Si la séquence dépasse max_seq_len, elle est découpée en fenêtres
    glissantes avec overlap = stride, pour maximiser les exemples sur
    un petit corpus.

    Chaque item retourné : (input_ids, target_ids)
      input_ids  = seq[:-1]   (batch, seq_len - 1)
      target_ids = seq[1:]    (batch, seq_len - 1)
    """

    def __init__(
        self,
        snippets:   List[CodeSnippet],
        config:     ModelConfig,
        tokenizer:  LanguageTokenizer | None = None,
        stride:     int | None = None,
    ):
        self.config    = config
        self.tokenizer = tokenizer or LanguageTokenizer()
        self.stride    = stride or config.max_seq_len // 2

        self._sequences: List[torch.Tensor] = []
        for snippet in snippets:
            self._add_snippet(snippet.content)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]      # (seq_len,)
        return seq[:-1], seq[1:]        # input, target — décalage d'un token

    # ── Construction ──────────────────────────────────────────────────────────

    def _add_snippet(self, source: str) -> None:
        ids = self.tokenizer.encode(source, add_special_tokens=True)
        # [BOS, t1, ..., tn, EOS] — EOS présent, on veut que le modèle
        # apprenne à le prédire en fin de programme

        seq = torch.tensor(ids, dtype=torch.long)
        max_len = self.config.max_seq_len

        if len(seq) <= max_len:
            self._sequences.append(seq)
        else:
            # Fenêtres glissantes
            start = 0
            while start + 1 < len(seq):
                end = min(start + max_len, len(seq))
                self._sequences.append(seq[start:end])
                if end == len(seq):
                    break
                start += self.stride