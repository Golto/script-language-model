import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class CodeSnippet:
    name:        str
    description: str
    signature:   str        # brut : "[x: float] -> float"
    content:     str        # code source, stripped


# ─── Parser ───────────────────────────────────────────────────────────────────

# // name: description signature
_HEADER_RE = re.compile(
    r'^//\s+'           # délimiteur
    r'(\w+)'            # name
    r':\s+'             # :
    r'(.+?)'            # description (non-greedy)
    r'\s+'              # espace avant signature
    r'(\[.*?\]\s*->.*?' # signature : [params] -> type
    r')\s*$'
)


class SnippetParseError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"Ligne {line} : {message}")
        self.line = line


def parse_snippets(text: str) -> List[CodeSnippet]:
    """Parse un fichier de snippets et retourne la liste des CodeSnippet."""
    snippets: List[CodeSnippet] = []
    current_header: re.Match | None = None
    current_header_line: int = 0
    body_lines: List[str] = []

    lines = text.splitlines()

    def _flush(until_line: int):
        """Finalise le snippet courant."""
        if current_header is None:
            return
        content = '\n'.join(body_lines).strip()
        if not content:
            raise SnippetParseError(
                f"Snippet '{current_header.group(1)}' sans contenu",
                current_header_line
            )
        snippets.append(CodeSnippet(
            name        = current_header.group(1).strip(),
            description = current_header.group(2).strip(),
            signature   = current_header.group(3).strip(),
            content     = content,
        ))

    for lineno, line in enumerate(lines, start=1):
        if line.startswith('//'):
            _flush(lineno)
            match = _HEADER_RE.match(line)
            if not match:
                raise SnippetParseError(f"En-tête invalide : '{line}'", lineno)
            current_header = match
            current_header_line = lineno
            body_lines = []
        else:
            if current_header is not None:
                body_lines.append(line)

    _flush(len(lines))
    return snippets


def parse_snippet_file(path: str | Path) -> List[CodeSnippet]:
    """Charge et parse un fichier de snippets depuis le disque."""
    return parse_snippets(Path(path).read_text(encoding='utf-8'))