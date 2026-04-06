from typing import List, Optional, Tuple

from src.data.snippet import CodeSnippet, parse_snippet_file
from src.data.specification import (
    ProgramSpecification,
    build_specification,
    SpecParseError,
)
from src.model.config import ModelConfig
from src.tokenizer import LanguageTokenizer
from .base import BaseDataset


class InstructDataset(BaseDataset):
    """
    Dataset pour l'entraînement instruct : BOS + spec + code + EOS.

    Chaque snippet est associé à une ProgramSpecification construite
    automatiquement depuis sa signature et son contenu.

    Les snippets dont la signature est invalide ou non-parsable sont
    silencieusement ignorés (logged via skip_count).
    """

    def __init__(
        self,
        snippets:  List[CodeSnippet],
        config:    ModelConfig,
        tokenizer: Optional[LanguageTokenizer] = None,
        n_examples: int = 2,
        seed:       Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, seed=seed, **kwargs)
        self.n_examples  = n_examples
        self.skip_count  = 0

        for snippet in snippets:
            self._add_snippet(snippet, seed=seed)

    # ── Interface BaseDataset ─────────────────────────────────────────────────

    def _encode(self, source: str) -> List[int]:
        """
        Non utilisé directement : l'encodage instruct requiert une spécification.
        Implémenté pour satisfaire l'interface BaseDataset.
        """
        return self.tokenizer.encode(source, add_special_tokens=True)

    def _augment_source(self, source: str) -> None:
        """
        Désactivé
        """
        pass


    # ── Ajout avec spécification ──────────────────────────────────────────────

    def _add_snippet(self, snippet: CodeSnippet, seed: Optional[int] = None) -> None:
        try:
            spec = build_specification(snippet, n_examples=self.n_examples, seed=seed)
        except SpecParseError:
            self.skip_count += 1
            return

        self._add_instruct(spec, snippet.content)

    
    def _add_instruct(self, spec: ProgramSpecification, source: str) -> None:
        """Encode et ajoute la paire (spec, source) + variantes augmentées."""
        self._push(self._encode_instruct(spec, source))

        if not self.augment:
            return

        # TODO data augmentation


    def _encode_instruct(self, spec: ProgramSpecification, source: str) -> List[int]:
        return self.tokenizer.encode_instruct(spec, source)
    
    
    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        snippet_files: List[str],
        config:        ModelConfig,
        **kwargs,
    ) -> "InstructDataset":
        snippets = []
        for path in snippet_files:
            snippets.extend(parse_snippet_file(path))
        if not snippets:
            raise ValueError("Aucun snippet chargé.")
        dataset = cls(snippets, config, **kwargs)
        if dataset.skip_count > 0:
            print(f"  [InstructDataset] {dataset.skip_count} snippet(s) ignorés "
                  f"(signature invalide ou non-parsable)")
        return dataset