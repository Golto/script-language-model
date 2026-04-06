from typing import List

from src.language.evaluator.environment import ValidRegisterType
from src.data.specification import ProgramSpecification
from .vocabulary import SPEC_START_TOKEN, SPEC_END_TOKEN


# ─── Encodage ─────────────────────────────────────────────────────────────────

def _value_to_tokens(value: ValidRegisterType) -> List[str]:
    """
    Convertit une valeur Python en liste de tokens du vocab.
    Cohérent avec la tokenisation du code :
      -5.0  → ['-', '5', '.', '0']
      True  → ['true']
      42    → ['4', '2']
    """
    if isinstance(value, bool):
        return ["true" if value else "false"]

    tokens: List[str] = []
    if isinstance(value, (int, float)) and value < 0:
        tokens.append("-")
        value = abs(value)

    # chiffre par chiffre + '.'
    s = str(int(value)) if isinstance(value, int) else str(float(value))
    tokens.extend(list(s))
    return tokens


def encode_specification(spec: ProgramSpecification) -> List[str]:
    """
    Retourne la séquence de tokens (strings) de la spec, BOS/EOS exclus.

    Format produit :
        ```
        <|specification>
        input-type r0 float
        input-type r1 float
        
        output-type float
        
        example 1 . 0 - 5 . 0 return 4 . 0
        example 2 . 1 3 . 7 return 5 . 8
        <specification|>
        ```
    """
    tokens: List[str] = [SPEC_START_TOKEN, "\n"]

    # Inputs
    for reg, typ in spec.inputs:
        tokens += ["input-type", reg, typ, "\n"]

    tokens.append("\n")

    # Outputs
    for typ in spec.output_types:
        tokens += ["output-type", typ, "\n"]

    # Exemples
    if spec.examples:
        tokens.append("\n")
        for ex in spec.examples:
            tokens.append("example")
            for v in ex.inputs:
                tokens.extend(_value_to_tokens(v))
            tokens.append("return")
            for v in ex.outputs:
                tokens.extend(_value_to_tokens(v))
            tokens.append("\n")

    tokens += [SPEC_END_TOKEN, "\n"]
    return tokens