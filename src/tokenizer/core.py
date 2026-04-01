from typing import List
from src.language.lexer import Lexer
from src.language.lexer.tokens import TokenType
from .vocabulary import (
    TOKEN_TO_ID, ID_TO_TOKEN, 
    VOCAB_SIZE, 
    BOS_TOKEN, EOS_TOKEN,
    BOS_ID, EOS_ID, PAD_ID
)





class UnknownTokenError(Exception):
    def __init__(self, value: str):
        super().__init__(f"Token inconnu dans le vocabulaire : '{value}'")


class LanguageTokenizer:
    """
    Tokenizer pour le langage d'embedding.
    Encode un programme source en séquence d'ids entiers.
    Les nombres sont décomposés chiffre par chiffre (+ point décimal).
    """

    VOCAB_SIZE = VOCAB_SIZE
    BOS_ID     = BOS_ID
    EOS_ID     = EOS_ID
    PAD_ID     = PAD_ID
    DIGIT_CHARS = set('0123456789.')

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, source: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._lex(source)
        ids = []
        for tok in tokens:
            ids.extend(self._token_to_ids(tok))
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids


    def _lex(self, source: str) -> List[str]:
        lexer = Lexer(source)
        raw = []
        for token in lexer.tokenize():
            if token.type == TokenType.EOF:
                continue
            if token.type == TokenType.NEWLINE:
                raw.append('\n')
                continue
            raw.append(token.value)
        return raw


    def _token_to_ids(self, value: str) -> List[int]:
        """
        Convertit une valeur lexicale en liste d'ids.
        Les nombres (INTEGER / FLOAT) sont décomposés caractère par caractère.
        """
        # Nombre : décompose en chiffres et '.'
        if all(c in '0123456789.' for c in value):
            ids = []
            for ch in value:
                if ch not in TOKEN_TO_ID:
                    raise UnknownTokenError(ch)
                ids.append(TOKEN_TO_ID[ch])
            return ids

        # Token atomique
        if value not in TOKEN_TO_ID:
            raise UnknownTokenError(value)
        return [TOKEN_TO_ID[value]]


    # ── Decode ────────────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> List[str]:
        result = []
        for i in ids:
            if i not in ID_TO_TOKEN:
                raise UnknownTokenError(f"id={i}")
            tok = ID_TO_TOKEN[i]
            if skip_special_tokens and tok in (BOS_TOKEN, EOS_TOKEN):
                continue
            result.append(tok)
        return result

    def decode_str(self, ids: list[int]) -> str:
        """Retourne une représentation lisible (espace entre tokens)."""
        tokens = self.decode(ids)
        parts = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '\n':
                # Retire l'espace traînant éventuel avant le \n
                if parts and parts[-1] == ' ':
                    parts.pop()
                parts.append('\n')
                i += 1
                continue

            # Chiffres/point : on colle tout le groupe numérique
            if token in self.DIGIT_CHARS:
                num = ''
                while i < len(tokens) and tokens[i] in self.DIGIT_CHARS:
                    num += tokens[i]
                    i += 1
                parts.append(num)
            else:
                parts.append(token)
                i += 1

            # Espace séparateur (sauf si le prochain est \n)
            if i < len(tokens) and tokens[i] != '\n':
                parts.append(' ')

        return ''.join(parts)