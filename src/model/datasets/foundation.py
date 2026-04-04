from typing import List, Optional
from src.data.snippet import CodeSnippet, parse_snippet_file
from src.model.config import ModelConfig
from src.tokenizer import LanguageTokenizer
from .base import BaseDataset


class FoundationDataset(BaseDataset):
    """Dataset pour l'entraînement foundation : BOS + code + EOS."""

    def __init__(
        self,
        snippets:  List[CodeSnippet],
        config:    ModelConfig,
        tokenizer: Optional[LanguageTokenizer] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, **kwargs)
        for snippet in snippets:
            self.add(snippet.content)

    def _encode(self, source: str) -> List[int]:
        return self.tokenizer.encode(source, add_special_tokens=True)

    @classmethod
    def from_files(
        cls,
        snippet_files: List[str],
        config:        ModelConfig,
        **kwargs,
    ) -> "FoundationDataset":
        snippets = []
        for path in snippet_files:
            snippets.extend(parse_snippet_file(path))
        if not snippets:
            raise ValueError("Aucun snippet chargé.")
        return cls(snippets, config, **kwargs)