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


    def tokens_to_source(
        self,
        tokens: List[str],
        indent: Optional[int] = 4,
    ) -> str:
        """
        Reconstruit un programme source lisible depuis une liste de tokens strings.

        Règles :
        - Les chiffres et '.' consécutifs sont collés en nombre
        - Le '-' précédant immédiatement un groupe numérique est un signe unaire
        si le token précédent est un opérateur, délimiteur ou mot-clé
        - Les sauts de ligne '\n' sont émis tels quels
        - L'indentation est gérée par les tokens then/do/else/endif/endwhile
        - Un espace est ajouté entre les tokens sauf avant '\n'
        """
        DIGIT_CHARS = self.DIGIT_CHARS  # set('0123456789.')

        # Tokens qui induisent un '-' unaire (pas un opérateur binaire)
        UNARY_TRIGGERS = {
            '=', '(', '+', '-', '*', '/', '%',
            '==', '!=', '<', '>', '<=', '>=',
            'and', 'or', 'not',
            'then', 'do', 'else', 'if', 'while',
            'input', 'output', 'return',
            '\n', None,   # début de ligne ou début de séquence
        }

        # Tokens de spec où '-' est toujours unaire (pas d'expression binaire)
        SPEC_LINE_STARTERS = {'example', 'input-type', 'output-type'}

        parts: List[str] = []
        current_indent   = 0
        is_new_line      = True
        in_spec_line     = False
        prev_token: Optional[str] = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # ── Saut de ligne ─────────────────────────────────────────────────────
            if tok == '\n':
                # Retire l'espace traînant
                if parts and parts[-1] == ' ':
                    parts.pop()
                parts.append('\n')
                is_new_line  = True
                in_spec_line = False
                prev_token   = '\n'
                i += 1
                continue

            # ── Indentation décroissante (avant écriture) ─────────────────────────
            if indent is not None and tok in ('endif', 'endwhile', 'else'):
                current_indent = max(0, current_indent - 1)

            # ── Indentation en début de ligne ─────────────────────────────────────
            if indent is not None and is_new_line:
                parts.append(' ' * (current_indent * indent))
                is_new_line = False

            # ── Contexte spec (exemple : ligne 'example ...' ou 'input-type ...') ─
            if tok in SPEC_LINE_STARTERS:
                in_spec_line = True

            # ── Détection signe unaire ────────────────────────────────────────────
            is_unary = (
                tok == '-'
                and (i + 1) < len(tokens)
                and tokens[i + 1] in DIGIT_CHARS - {'.'}  # suivi d'un chiffre
                and (in_spec_line or prev_token in UNARY_TRIGGERS)
            )

            # ── Nombre (éventuellement précédé d'un signe unaire) ─────────────────
            if tok in DIGIT_CHARS or is_unary:
                num = ''
                if is_unary:
                    num += '-'
                    i += 1

                has_dot = False
                while i < len(tokens) and tokens[i] in DIGIT_CHARS:
                    c = tokens[i]
                    if c == '.':
                        if has_dot:
                            break      # deuxième point → nouveau nombre, on arrête
                        has_dot = True
                    num += c
                    i += 1

                parts.append(num)
                prev_token = num  # le nombre entier comme token "précédent"

            # ── Token ordinaire ───────────────────────────────────────────────────
            else:
                parts.append(tok)
                prev_token = tok
                i += 1

            # ── Indentation croissante (après écriture) ───────────────────────────
            if indent is not None and tok in ('then', 'do', 'else'):
                current_indent += 1

            # ── Espace séparateur (sauf avant '\n') ───────────────────────────────
            if i < len(tokens) and tokens[i] != '\n':
                parts.append(' ')

        return ''.join(parts)