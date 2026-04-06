from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

from src.data.snippet import CodeSnippet
from src.language.interpreter import Interpreter
from src.language.evaluator.environment import ValidRegisterType

# ─── Types ────────────────────────────────────────────────────────────────────

InputType  = Literal["int", "float", "bool"]
OutputType = Literal["int", "float", "bool", "number"]

_VALID_INPUT_TYPES  = {"int", "float", "bool", "number"}
_VALID_OUTPUT_TYPES = {"int", "float", "bool", "number"}


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class SpecExample:
    inputs:  List[ValidRegisterType]
    outputs: List[ValidRegisterType]


@dataclass
class ProgramSpecification:
    """
    Spécification complète d'un programme.

    inputs      : liste ordonnée (registre, type) : ex: [("r0", "float"), ("r1", "float")]
    output_types: liste des types de sortie       : ex: ["float"]
                  (plusieurs si outputs multiples  : ex: ["bool", "float"])
    examples    : paires (inputs → outputs) générées par exécution
    """
    inputs:       List[Tuple[str, InputType]]
    output_types: List[OutputType]
    examples:     List[SpecExample] = field(default_factory=list)


# ─── Parser de signature ──────────────────────────────────────────────────────

# [a: float, b: float] -> float
# [a: float, b: float] -> (bool, float)
# [] -> int
_SIG_RE = re.compile(
    r'\[(?P<params>[^\]]*)\]'   # [params]
    r'\s*->\s*'
    r'(?P<ret>.+)$'
)
_PARAM_RE  = re.compile(r'(\w+)\s*:\s*(\w+)')
_MULTI_RET = re.compile(r'\(([^)]+)\)')


def _normalize_type(t: str) -> str:
    t = t.strip().lower()
    if t in ("integer",):
        return "int"
    if t in ("boolean",):
        return "bool"
    return t


def parse_signature(signature: str) -> Tuple[List[InputType], List[OutputType]]:
    """
    Parse une signature brute en (input_types, output_types).

    Exemples :
      "[a: float, b: float] -> float"      → (["float", "float"], ["float"])
      "[x: int] -> (bool, float)"          → (["int"],            ["bool", "float"])
      "[] -> int"                          → ([],                 ["int"])
    """
    m = _SIG_RE.match(signature.strip())
    if not m:
        raise ValueError(f"Signature invalide : '{signature}'")

    # Inputs
    input_types: List[InputType] = []
    for _, typ in _PARAM_RE.findall(m.group("params")):
        t = _normalize_type(typ)
        if t not in _VALID_INPUT_TYPES:
            raise ValueError(f"Type d'input inconnu : '{t}'")
        input_types.append(t)

    # Outputs
    ret_str = m.group("ret").strip()
    multi   = _MULTI_RET.match(ret_str)
    if multi:
        raw_types = [s.strip() for s in multi.group(1).split(",")]
    else:
        raw_types = [ret_str]

    output_types: List[OutputType] = []
    for t in raw_types:
        t = _normalize_type(t)
        if t not in _VALID_OUTPUT_TYPES:
            raise ValueError(f"Type d'output inconnu : '{t}'")
        output_types.append(t)

    return input_types, output_types


# ─── Inférence des registres depuis le code ───────────────────────────────────

def _infer_input_registers(source: str) -> List[str]:
    """
    Retourne les registres lus par 'input rX', dans l'ordre d'apparition.
    Ne considère que les input en dehors d'un bloc if/while (niveau 0).
    Approximation conservatrice : on prend tous les 'input rX' dans l'ordre.
    """
    return re.findall(r'^\s*input\s+(r\d+)', source, re.MULTILINE)


# ─── Génération de valeurs aléatoires typées ─────────────────────────────────

def _random_value(typ: InputType, rng: random.Random) -> ValidRegisterType:
    if typ == "bool":
        return rng.choice([True, False])
    if typ == "int":
        return rng.choice([-5, -3, -1, 0, 1, 2, 3, 5, 7, 10])
    if typ in ("float", "number"):
        base = rng.choice([-3.0, -1.5, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
        # return round(base + rng.uniform(-0.5, 0.5), 2)
        return base
    return 1


# ─── Exécution pour générer des exemples ─────────────────────────────────────

def _run_with_inputs(
    source: str,
    values: List[ValidRegisterType],
) -> Optional[List[ValidRegisterType]]:
    it = iter(values)
    outputs: List[ValidRegisterType] = []

    def input_fn(_reg: str) -> ValidRegisterType:
        try:
            return next(it)
        except StopIteration:
            return 0

    def output_fn(value: ValidRegisterType) -> None:
        if isinstance(value, float):
            # fix: notation scientifique x.xe+-yy n'est pas tokenisable
            #      arrondi pour l'éviter
            value = round(value, 4)

            # pour éviter un "-0.0"
            if value == 0.0:
                value = 0.0 
        
        outputs.append(value)

    try:
        Interpreter().execute(source, input_fn=input_fn, output_fn=output_fn)
        return outputs if outputs else None
    except Exception:
        return None


def _generate_examples(
    source:      str,
    input_types: List[InputType],
    n:           int = 3,
    max_tries:   int = 20,
    seed:        int | None = None,
) -> List[SpecExample]:
    """
    Génère n exemples valides en tirant des inputs aléatoires typés.
    Essaie de produire des exemples avec des outputs distincts.
    """
    rng      = random.Random(seed)
    seen     : set = set()
    examples : List[SpecExample] = []

    for _ in range(max_tries):
        if len(examples) >= n:
            break

        values  = [_random_value(t, rng) for t in input_types]
        outputs = _run_with_inputs(source, values)

        if outputs is None:
            continue

        # Déduplique sur le couple (inputs, outputs)
        key = (tuple(values), tuple(outputs))
        if key in seen:
            continue
        seen.add(key)

        examples.append(SpecExample(inputs=values, outputs=outputs))

    return examples


# ─── API principale ───────────────────────────────────────────────────────────

class SpecParseError(Exception):
    pass


def build_specification(
    snippet:    CodeSnippet,
    n_examples: int = 2,
    seed:       int | None = None,
) -> ProgramSpecification:
    """
    Construit une ProgramSpecification complète depuis un CodeSnippet :
      1. Parse la signature pour obtenir les types
      2. Infère les registres depuis le code source
      3. Génère n_examples exemples par exécution

    Lève SpecParseError si la signature est invalide ou si les registres
    ne correspondent pas aux types déclarés.
    """
    try:
        input_types, output_types = parse_signature(snippet.signature)
    except ValueError as e:
        raise SpecParseError(str(e)) from e

    registers = _infer_input_registers(snippet.content)

    # Aligner registres et types : tronque au minimum des deux
    if len(registers) != len(input_types):
        # Accepte la divergence silencieusement : on aligne sur le min
        n = min(len(registers), len(input_types))
        registers   = registers[:n]
        input_types = input_types[:n]

    inputs = list(zip(registers, input_types))

    examples = _generate_examples(
        source      = snippet.content,
        input_types = input_types,
        n           = n_examples,
        seed        = seed,
    )

    return ProgramSpecification(
        inputs       = inputs,
        output_types = output_types,
        examples     = examples,
    )