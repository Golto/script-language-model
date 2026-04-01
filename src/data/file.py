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
    char: str = "g"
) -> None:
    """Écrit un fichier snippet_gXXXX ."""
    if len(char) > 1:
        raise ValueError("`char` doit être un caractère.")
    filename = output_dir / f"snippet_{char}{file_index:04d}"
    lines = []
    for i, s in enumerate(snippets):
        lines.append(f"// {s.name}: {s.description} {s.signature}")
        lines.append(s.content)
    filename.write_text("\n".join(lines), encoding="utf-8")