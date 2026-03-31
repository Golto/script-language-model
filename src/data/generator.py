from __future__ import annotations

import re
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Tuple, Optional

import ollama

from src.language import Interpreter
from src.language.evaluator.environment import ValidRegisterType
from src.language.lexer.core import LanguageLexicalError
from src.language.parser.base import LanguageSyntaxicalError
from src.language.evaluator.exception import LanguageExecutionError
from src.data.snippet import parse_snippets, CodeSnippet, SnippetParseError


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    model: str              = "qwen3.5:2b-q4_K_M"
    snippets_per_batch: int = 5
    max_retries: int        = 3
    temperature: float      = 0.9
    max_new_tokens: int     = 1024
    output_dir: str         = ".private/data/snippets"
    snippets_per_file: int  = 50
    start_index: int        = 0


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a code generator for a minimal register-based programming language.

LANGUAGE RULES:
- 16 registers: r0 to r15 (the only variables)
- Types: integers (e.g. 42), floats (e.g. 3.14), booleans (true / false)
- Arithmetic: + - * / %
- Comparison: == != < > <= >=
- Logic: and or not
- Assignment: r0 = <expr>
- Conditionals: if <bool_expr> then ... [else ...] endif
- Loops: while <bool_expr> do ... endwhile  (supports break / continue)
- I/O: input r0   (read into register) / output <expr>  (print)
- One instruction per line. No functions, no strings, no arrays.

OUTPUT FORMAT — strictly follow this, no extra text:
// name: Short description [param: type, ...] -> return_type
<code using only the syntax above>

// name2: Short description [param: type, ...] -> return_type
<code>

EXAMPLES:
// cube: Cubing a number [x: float] -> float
input r0
output r0 * r0 * r0

// max3: Maximum of three numbers [a: float, b: float, c: float] -> float
input r0
input r1
input r2
r3 = r0
if r1 > r3 then
    r3 = r1
endif
if r2 > r3 then
    r3 = r2
endif
output r3

// gcd: Greatest common divisor [a: int, b: int] -> int
input r0
input r1
while r1 > 0 do
    r2 = r0 % r1
    r0 = r1
    r1 = r2
endwhile
output r0

NOTE:
Do not reproduce examples
"""

_THEMES = [
    "arithmetic and math utilities",
    "comparisons and min/max operations",
    "iterative algorithms (sum, product, factorial)",
    "boolean logic and conditionals",
    "number properties (even, odd, positive, negative)",
    "simple approximations (power, integer square root)",
    "input validation and clamping",
    "counting and accumulation loops",
]

def _make_user_prompt(n: int, theme: str) -> str:
    return (
        f"Generate exactly {n} different snippets about: {theme}.\n"
        f"Follow the format strictly. No explanation, no markdown, only the snippets."
    )


# ─── Validation ───────────────────────────────────────────────────────────────

def _silent_input(register_name: str) -> ValidRegisterType:
    """Input factice pour la validation, retourne 1 pour tout registre."""
    return 1

def _silent_output(value: ValidRegisterType) -> None:
    pass


def validate_snippet(snippet: CodeSnippet) -> Tuple[bool, Optional[str]]:
    """
    Tente d'exécuter le snippet avec des inputs factices.
    Retourne (True, None) si valide, (False, raison) sinon.
    """
    interpreter = Interpreter()
    try:
        interpreter.execute(
            snippet.content,
            input_fn=_silent_input,
            output_fn=_silent_output,
        )
        return True, None
    except (LanguageLexicalError, LanguageSyntaxicalError, LanguageExecutionError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Erreur inattendue: {e}"
    

# ─── Génération ───────────────────────────────────────────────────────────────

def _generate_batch(
    config: GeneratorConfig,
    theme: str,
) -> List[CodeSnippet]:
    """
    Appelle Ollama, parse la réponse, retourne les snippets parsés.
    Lève ValueError si le parse échoue après max_retries.
    """
    prompt = _make_user_prompt(config.snippets_per_batch, theme)

    for attempt in range(1, config.max_retries + 1):
        response = ollama.chat(
            model=config.model,
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
            options={
                "num_predict": config.max_new_tokens,
                "temperature": config.temperature
            },
            think=False
        )
        raw = response["message"]["content"]

        try:
            snippets = parse_snippets(raw)
            if snippets:
                return snippets
        except SnippetParseError:
            pass

    return []


def generate_snippets(
    config: GeneratorConfig | None = None,
    total: int = 50,
    on_accept: Callable[[CodeSnippet, int], None] | None = None,
    on_reject: Callable[[CodeSnippet, str], None]  | None = None,
) -> list[CodeSnippet]:
    """
    Génère `total` snippets valides et les écrit dans output_dir.

    on_accept(snippet, global_index) — callback à chaque snippet accepté
    on_reject(snippet, reason)       — callback à chaque snippet rejeté
    """
    config     = config or GeneratorConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted:    List[CodeSnippet] = []
    file_buffer: List[CodeSnippet] = []
    file_index   = config.start_index // config.snippets_per_file
    global_index = config.start_index
    themes       = _THEMES.copy()

    while len(accepted) < total:
        theme = random.choice(themes)
        batch = _generate_batch(config, theme)

        for snippet in batch:
            if len(accepted) >= total:
                break

            ok, reason = validate_snippet(snippet)

            if ok:
                accepted.append(snippet)
                file_buffer.append(snippet)
                if on_accept:
                    on_accept(snippet, global_index)
                global_index += 1

                # Flush vers fichier quand le buffer est plein
                if len(file_buffer) >= config.snippets_per_file:
                    _flush_file(file_buffer, file_index, output_dir)
                    file_buffer = []
                    file_index += 1
            else:
                if on_reject:
                    on_reject(snippet, reason or "inconnu")

    # Flush le reste
    if file_buffer:
        _flush_file(file_buffer, file_index, output_dir)

    return accepted


def _flush_file(
    snippets: List[CodeSnippet],
    file_index: int,
    output_dir: Path,
) -> None:
    """Écrit un fichier snippet_gXXXX ."""
    filename = output_dir / f"snippet_g{file_index:04d}"
    lines = []
    for i, s in enumerate(snippets):
        if i > 0:
            lines.append("")   # ligne vide entre snippets
        lines.append(f"// {s.name}: {s.description} {s.signature}")
        lines.append(s.content)
    filename.write_text("\n".join(lines), encoding="utf-8")