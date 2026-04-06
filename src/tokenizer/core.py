from typing import List, Optional
from src.language.lexer import Lexer
from src.language.lexer.tokens import TokenType
from src.data.specification import ProgramSpecification
from .vocabulary import (
    TOKEN_TO_ID, ID_TO_TOKEN, 
    VOCAB_SIZE, 
    BOS_TOKEN, EOS_TOKEN,
    BOS_ID, EOS_ID, PAD_ID,
    SPEC_START_TOKEN, SPEC_END_TOKEN,
)
from .specification import encode_specification as _encode_spec


class UnknownTokenError(Exception):
    def __init__(self, value: str):
        super().__init__(f"Token inconnu dans le vocabulaire : '{value}'")


class LanguageTokenizer:
    """
    Tokenizer pour le langage d'embedding.
    Encode un programme source en séquence d'ids entiers.
    Les nombres sont décomposés chiffre par chiffre (+ point décimal).

    Pour deux types de modèles:
    - completion : .encode()         : BOS code EOS
    - instruct   : .ecode_instruct() : BOS spec code EOS
    """

    VOCAB_SIZE = VOCAB_SIZE
    BOS_ID     = BOS_ID
    EOS_ID     = EOS_ID
    PAD_ID     = PAD_ID
    DIGIT_CHARS = set('0123456789.')

    # ── Foundation encode ─────────────────────────────────────────────────────

    def encode(self, source: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self._lex(source)
        ids = []
        for tok in tokens:
            ids.extend(self._token_to_ids(tok))
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids
    

    # ── Instruct encode ───────────────────────────────────────────────────────

    def encode_instruct(
        self,
        specification: ProgramSpecification,
        source: Optional[str] = None,
    ) -> List[int]:
        """
        Encode une séquence instruct complète.

        À l'entraînement : source fourni   BOS spec code EOS
        À l'inférence    : source=None     BOS spec           (le modèle complète)
        """
        spec_ids = self._spec_to_ids(specification)

        if source is None:
            return [BOS_ID] + spec_ids

        code_ids = self.encode(source, add_special_tokens=False)
        return [BOS_ID] + spec_ids + code_ids + [EOS_ID]


    def encode_spec_only(self, specification: ProgramSpecification) -> List[int]:
        """Encode uniquement la spec (sans BOS/EOS ni code). Utile pour debug."""
        return self._spec_to_ids(specification)


    def _spec_to_ids(self, specification: ProgramSpecification) -> List[int]:
        """Convertit une ProgramSpec en liste d'ids via encode_spec."""
        tokens = _encode_spec(specification)
        ids = []
        for tok in tokens:
            if tok not in TOKEN_TO_ID:
                raise UnknownTokenError(tok)
            ids.append(TOKEN_TO_ID[tok])
        return ids
    

    # ── Lexer / token_to_ids ──────────────────────────────────────────────────

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
        _SPECIAL = {BOS_TOKEN, EOS_TOKEN, SPEC_START_TOKEN, SPEC_END_TOKEN}
        result = []
        for i in ids:
            if i not in ID_TO_TOKEN:
                raise UnknownTokenError(f"id={i}")
            tok = ID_TO_TOKEN[i]
            if skip_special_tokens and tok in _SPECIAL:
                continue
            result.append(tok)
        return result

    def decode_str(
        self, 
        ids: List[int], 
        skip_special_tokens: bool = True, 
        indent: Optional[int] = None
    ) -> str:
        """
        Retourne une représentation lisible (espace entre tokens).
        Si `indent` est fourni (ex: 4), le code sera automatiquement indenté.
        """
        tokens = self.decode(ids, skip_special_tokens=skip_special_tokens)
        parts = []
        index = 0

        current_indent = 0
        is_new_line = True

        while index < len(tokens):
            token = tokens[index]

            # Saut de ligne
            if token == '\n':
                if parts and parts[-1] == ' ':
                    parts.pop()
                parts.append('\n')
                is_new_line = True
                index += 1
                continue
            
            # 1. Ajuster l'indentation (diminution)
            if indent is not None:
                if token in ('endif', 'endwhile', 'else'):
                    current_indent = max(0, current_indent - 1)
                
                if is_new_line:
                    parts.append(' ' * (current_indent * indent))
                    is_new_line = False

            # 2. Reconstruire les nombres / Ajouter le token courant
            if token in self.DIGIT_CHARS:
                num = ''
                while index < len(tokens) and tokens[index] in self.DIGIT_CHARS:
                    num += tokens[index]
                    index += 1
                parts.append(num)
            else:
                parts.append(token)
                index += 1

            # 3. Ajuster l'indentation (augmentation)
            if indent is not None:
                if token in ('then', 'do', 'else'):
                    current_indent += 1

            # 4. Ajouter un espace d'espacement si le prochain n'est pas un saut de ligne
            if index < len(tokens) and tokens[index] != '\n':
                parts.append(' ')

        return ''.join(parts)