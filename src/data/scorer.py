import re
from typing import List, Optional, Tuple, Callable, Set

from src.data.snippet import CodeSnippet
from src.data.file import validate_snippet
from src.language.interpreter import Interpreter
from src.language.evaluator.environment import ValidRegisterType


SnippetScorer = Callable[[CodeSnippet], bool]


# ─── Helpers d'exécution ──────────────────────────────────────────────────────

def _run(source: str, inputs: List[ValidRegisterType]) -> Optional[List[ValidRegisterType]]:
    it = iter(inputs)
    outputs = []

    def input_fn(_reg: str) -> ValidRegisterType:
        try:
            return next(it)
        except StopIteration:
            return 0

    def output_fn(value: ValidRegisterType) -> None:
        outputs.append(value)

    try:
        Interpreter().execute(source, input_fn=input_fn, output_fn=output_fn)
        return outputs
    except Exception:
        return None


# ─── Analyse statique ─────────────────────────────────────────────────────────

def _count_inputs(source: str) -> int:
    return len(re.findall(r'^\s*input\s+r\d+', source, re.MULTILINE))

def _count_outputs(source: str) -> int:
    return len(re.findall(r'^\s*output\b', source, re.MULTILINE))

def _program_length(source: str) -> int:
    return len([l for l in source.splitlines() if l.strip()])

def _count_unique_registers(source: str) -> int:
    return len(set(re.findall(r'\br\d+\b', source)))

def _has_control_flow(source: str) -> dict:
    return {
        "if":       bool(re.search(r'\bif\b',       source)),
        "while":    bool(re.search(r'\bwhile\b',    source)),
        "break":    bool(re.search(r'\bbreak\b',    source)),
        "continue": bool(re.search(r'\bcontinue\b', source)),
    }

def _used_operators(source: str) -> Set[str]:
    return set(re.findall(r'(?<![=!<>])(?:==|!=|<=|>=|[+\-*/%<>])', source))


# ─── Disqualifiants statiques ─────────────────────────────────────────────────

# x op x : même opérande des deux côtés
_TRIVIAL_BINOP_RE = re.compile(
    r'\b(r\d+|\d+(?:\.\d+)?)\s*(==|!=|<|>|<=|>=|-|\+)\s*\1\b'
)

# x = x : assignation inutile
_SELF_ASSIGN_RE = re.compile(
    r'\b(r\d+)\s*=\s*\1\b'
)

# Comparaison entre deux littéraux : "3 > 2", résultat connu statiquement
_CONST_CMP_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(==|!=|<|>|<=|>=)\s*\d+(?:\.\d+)?\b'
)

# While dont la condition est un littéral booléen
_CONST_WHILE_RE = re.compile(r'\bwhile\s+(true|false)\s+do\b')

# If dont la condition est un littéral booléen
_CONST_IF_RE = re.compile(r'\bif\s+(true|false)\s+then\b')


def _has_trivial_binop(source: str) -> bool:
    return bool(_TRIVIAL_BINOP_RE.search(source))

def _has_self_assignment(source: str) -> bool:
    return bool(_SELF_ASSIGN_RE.search(source))

def _has_constant_condition(source: str) -> bool:
    """While ou if dont la condition est connue statiquement."""
    return bool(_CONST_WHILE_RE.search(source) or _CONST_IF_RE.search(source))

def _has_constant_comparison(source: str) -> bool:
    return bool(_CONST_CMP_RE.search(source))


# ─── Data-flow : variable définie mais jamais lue ────────────────────────────

def _defined_but_unused_registers(source: str) -> Set[str]:
    """
    Retourne l'ensemble des registres écrits (assignment ou input)
    mais jamais lus dans une expression.

    Approximation conservatrice : un registre est "lu" s'il apparaît
    dans une expression (droite d'un '=', condition, output, etc.)
    """
    lines = [l.strip() for l in source.splitlines() if l.strip()]

    written: Set[str] = set()
    read:    Set[str] = set()

    for line in lines:
        # input rX → écrit rX
        m = re.match(r'^input\s+(r\d+)$', line)
        if m:
            written.add(m.group(1))
            continue

        # rX = expr → écrit rX, lit tout ce qui est dans expr
        m = re.match(r'^(r\d+)\s*=\s*(.+)$', line)
        if m:
            written.add(m.group(1))
            read.update(re.findall(r'\br\d+\b', m.group(2)))
            continue

        # output expr, if expr, while expr → lecture pure
        read.update(re.findall(r'\br\d+\b', line))

    return written - read


def _has_unused_definitions(source: str) -> bool:
    return len(_defined_but_unused_registers(source)) > 0


# ─── Détection de structures triviales ───────────────────────────────────────

def _has_empty_loop_body(source: str) -> bool:
    """
    While dont le corps ne contient aucune instruction
    (ou uniquement des instructions qui ne touchent pas la condition).
    Heuristique : corps du while = 0 lignes non-vides entre do et endwhile.
    """
    body_re = re.compile(r'\bwhile\b.*?\bdo\b(.*?)\bendwhile\b', re.DOTALL)
    for m in body_re.finditer(source):
        body = m.group(1).strip()
        non_empty = [l for l in body.splitlines() if l.strip()]
        if len(non_empty) == 0:
            return True
    return False

def _while_condition_never_modified(source: str) -> bool:
    """
    Détecte un while dont le registre de condition n'est jamais modifié
    dans le corps : boucle infinie déguisée.
    Ex: while r0 do ... (r0 jamais assigné dans le corps)
    """
    body_re = re.compile(r'\bwhile\s+(r\d+)\s+do\b(.*?)\bendwhile\b', re.DOTALL)
    for m in body_re.finditer(source):
        cond_reg = m.group(1)
        body     = m.group(2)
        # Le registre de condition est-il écrit dans le corps ?
        if not re.search(rf'\b{cond_reg}\s*=', body):
            return True
    return False

def _has_no_accumulation(source: str) -> bool:
    """
    Détecte l'absence de pattern accumulateur : rX = rX op expr.
    Ce pattern est caractéristique des boucles non-triviales
    (factorielles, sommes, approximations).
    Retourne True si AUCUN accumulateur n'est détecté.
    """
    accum_re = re.compile(r'\b(r\d+)\s*=\s*\1\s*[+\-*/%]')
    return not bool(accum_re.search(source))


# ─── Test de variance I/O ─────────────────────────────────────────────────────

_INPUT_SETS = [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 2, 3, 4],
    [5, 3, 7, 2],
    [10, 0, 1, 5],
    [3, 3, 0, 1],
    [2, 5, 2, 8],
    [7, 1, 4, 3],
]

def _io_distinct_outputs(source: str) -> int:
    """Retourne le nombre d'outputs distincts sur l'ensemble des jeux de test."""
    seen = set()
    for inputs in _INPUT_SETS:
        result = _run(source, inputs)
        if result is not None:
            seen.add(tuple(result))
    return len(seen)

def _output_is_monotone(source: str) -> bool:
    """
    Vérifie si l'output croît (ou décroît) strictement avec le premier input.
    Caractéristique des fonctions mathématiques non-triviales.
    """
    results = []
    for v in [1, 2, 3, 4, 5]:
        out = _run(source, [v, v, v, v])
        if out and len(out) == 1 and isinstance(out[0], (int, float)):
            results.append(out[0])
    if len(results) < 3:
        return False
    diffs = [results[i+1] - results[i] for i in range(len(results)-1)]
    return all(d > 0 for d in diffs) or all(d < 0 for d in diffs)


# ─── Score principal ──────────────────────────────────────────────────────────

def score_snippet(snippet: CodeSnippet) -> Tuple[float, dict]:
    """
    Retourne (score ∈ [0.0, 1.0], détails).

    Disqualifiants (score = 0.0) :
      - exécution invalide
      - x op x  (binop trivial)
      - x = x   (self-assignment)
      - while/if avec condition booléenne littérale
      - corps de while vide
      - while dont la condition n'est jamais modifiée

    Pénalités :
      - pas d'output                     : -0.30
      - pas d'input                      : -0.15
      - output constant (0-1 distinct)   : -0.25
      - output indépendant des inputs    : -0.20
      - comparaisons entre littéraux     : -0.10
      - registres définis mais inutilisés: -0.10  (par registre, max -0.20)
      - while sans accumulateur          : -0.10

    Bonus :
      - présence de if                   : +0.08
      - présence de while                : +0.12
      - présence de break/continue       : +0.05
      - ≥ 2 registres distincts          : +0.05
      - programme ≥ 5 lignes             : +0.05
      - programme ≥ 10 lignes            : +0.05
      - ≥ 3 outputs distincts            : +0.08
      - ≥ 5 outputs distincts            : +0.08
      - output monotone                  : +0.10
      - opérateurs variés (≥ 3)          : +0.05
      - pattern accumulateur             : +0.10
    """
    source  = snippet.content
    details = {}

    # ── Disqualifiants ────────────────────────────────────────────────────────
    ok, reason = validate_snippet(snippet)
    if not ok:
        return 0.0, {"disqualified": f"execution_error: {reason}"}

    if _has_trivial_binop(source):
        return 0.0, {"disqualified": "trivial_binop (x op x)"}

    if _has_self_assignment(source):
        return 0.0, {"disqualified": "self_assignment (rX = rX)"}

    if _has_constant_condition(source):
        return 0.0, {"disqualified": "constant_condition (while/if true/false)"}

    if _has_empty_loop_body(source):
        return 0.0, {"disqualified": "empty_loop_body"}

    if _while_condition_never_modified(source):
        return 0.0, {"disqualified": "while_condition_never_modified"}

    # ── Base ──────────────────────────────────────────────────────────────────
    score = 0.4

    # I/O structure
    n_outputs = _count_outputs(source)
    n_inputs  = _count_inputs(source)
    details["n_outputs"] = n_outputs
    details["n_inputs"]  = n_inputs

    if n_outputs == 0:
        score -= 0.30
    if n_inputs == 0:
        score -= 0.15

    # Comparaisons constantes
    if _has_constant_comparison(source):
        details["constant_comparison"] = True
        score -= 0.10

    # Registres inutilisés
    unused = _defined_but_unused_registers(source)
    details["unused_registers"] = sorted(unused)
    score -= min(0.20, len(unused) * 0.10)

    # Variance I/O
    n_distinct = _io_distinct_outputs(source)
    details["distinct_outputs"] = n_distinct

    if n_inputs > 0:
        if n_distinct <= 1:
            score -= 0.25   # output constant
        if n_distinct <= 1:
            score -= 0.20   # output indépendant des inputs

        if n_distinct >= 3:
            score += 0.08
        if n_distinct >= 5:
            score += 0.08

    # Monotonie (bonus fort : caractéristique des fonctions mathématiques)
    if n_inputs > 0 and n_outputs > 0:
        monotone = _output_is_monotone(source)
        details["monotone"] = monotone
        if monotone:
            score += 0.10

    # ── Structure de contrôle ─────────────────────────────────────────────────
    cf = _has_control_flow(source)
    details["control_flow"] = cf

    if cf["if"]:
        score += 0.08
    if cf["while"]:
        score += 0.12
        # Pénalité while sans accumulateur
        no_accum = _has_no_accumulation(source)
        details["no_accumulation"] = no_accum
        if no_accum:
            score -= 0.10
        else:
            score += 0.10   # bonus accumulateur
    if cf["break"] or cf["continue"]:
        score += 0.05

    # ── Richesse du code ──────────────────────────────────────────────────────
    n_regs = _count_unique_registers(source)
    details["unique_registers"] = n_regs
    if n_regs >= 2:
        score += 0.05

    n_lines = _program_length(source)
    details["lines"] = n_lines
    if n_lines >= 5:
        score += 0.05
    if n_lines >= 10:
        score += 0.05

    ops = _used_operators(source)
    details["operators"] = sorted(ops)
    if len(ops) >= 3:
        score += 0.05

    return max(0.0, min(1.0, score)), details


# ─── Scorer binaire ───────────────────────────────────────────────────────────

def make_threshold_scorer(threshold: float = 0.6) -> SnippetScorer:
    def _scorer(snippet: CodeSnippet) -> bool:
        s, _ = score_snippet(snippet)
        return s >= threshold
    return _scorer