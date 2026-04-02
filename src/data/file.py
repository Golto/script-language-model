from typing import List, Tuple, Optional
from pathlib import Path

from src.language.interpreter import ValidRegisterType, Interpreter
from src.language.lexer import LanguageLexicalError
from src.language.parser import LanguageSyntaxicalError
from src.language.evaluator import LanguageExecutionError
from .snippet import CodeSnippet


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


# ─── Fichiers ─────────────────────────────────────────────────────────────────

def flush_file(
    snippets: List[CodeSnippet],
    file_index: int,
    output_dir: Path,
) -> None:
    """Écrit un fichier snippet_gXXXX ."""
    filename = output_dir / f"snippet_g{file_index:04d}"
    flush_snippets(snippets, filename)


def flush_snippets(snippets: List[CodeSnippet], path: Path) -> None:
    lines = []
    for i, s in enumerate(snippets):
        lines.append(f"// {s.name}: {s.description} {s.signature}")
        lines.append(s.content)
    path.write_text("\n".join(lines), encoding="utf-8")