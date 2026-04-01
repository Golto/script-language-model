import re
import random
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from src.tokenizer import LanguageTokenizer
from src.data.snippet import CodeSnippet

from .config import ModelConfig


# ─── Augmentation ─────────────────────────────────────────────────────────────

_ALL_REGISTERS = [f"r{i}" for i in range(16)]

_REG_RE = re.compile(r'\br(\d+)\b')

_COMPARISON_FLIP = {
    '>': '<', 
    '<': '>', 
    '>=': '<=', 
    '<=': '>=',
}


def _used_registers(source: str) -> List[str]:
    """Retourne la liste ordonnée des registres présents dans le source."""
    seen = []
    for m in _REG_RE.finditer(source):
        reg = m.group(0)
        if reg not in seen:
            seen.append(reg)
    return seen


def augment_registers(source: str, rng: random.Random) -> str:
    """
    Renomme les registres utilisés vers un sous-ensemble aléatoire disjoint.

    Exemple : source utilise r0, r1, r2
    → tirage sans remise de 3 registres parmi les 16 (ex: r7, r3, r11)
    → r0→r7, r1→r3, r2→r11 partout dans le source

    Garantit l'absence de conflits : le mapping est construit en une passe
    puis appliqué via un placeholder intermédiaire.
    """
    used = _used_registers(source)

    if not used:
        return source

    candidates = [r for r in _ALL_REGISTERS if r not in used]
    if len(candidates) < len(used):
        # Pas assez de registres libres pour un remap sans conflit
        return source

    targets = rng.sample(candidates, k=len(used))
    mapping = dict(zip(used, targets))

    # Passage en deux temps pour éviter les collisions transitoires
    # (ex: r0 -> r1, r1 -> r2 appiqué naïvement transformerait r0 en r2)
    # Étape 1 : placeholder unique par registre source
    placeholders = {reg: f"__REG{i}__" for i, reg in enumerate(used)}
    result = source
    for reg, ph in placeholders.items():
        result = re.sub(rf'\b{re.escape(reg)}\b', ph, result)

    # Étape 2 : placeholder -> registre cible
    for reg, ph in placeholders.items():
        result = result.replace(ph, mapping[reg])

    return result


def augment_flip_comparison(source: str, rng: random.Random) -> str:
    """Inverse aléatoirement une comparaison en swappant opérandes et opérateur."""
    # Matche: expr OP expr (simplifié pour registres et entiers)
    pattern = re.compile(r'(r\d+|\d+)\s*(>=|<=|>|<)\s*(r\d+|\d+)')
    
    matches = list(pattern.finditer(source))
    if not matches:
        return source
    
    m = rng.choice(matches)
    left, op, right = m.group(1), m.group(2), m.group(3)
    flipped = f"{right} {_COMPARISON_FLIP[op]} {left}"
    return source[:m.start()] + flipped + source[m.end():]

# ─── Dataset ──────────────────────────────────────────────────────────────────

class ProgramDataset(Dataset):
    """
    Dataset de séquences pour la prédiction du prochain token.

    Chaque snippet est encodé en [BOS, t1, ..., tn, EOS].
    Si la séquence dépasse max_seq_len, elle est découpée en fenêtres
    glissantes avec overlap = stride, pour maximiser les exemples sur
    un petit corpus.

    Chaque item retourné : (input_ids, target_ids)
      input_ids  = seq[:-1]   (batch, seq_len - 1)
      target_ids = seq[1:]    (batch, seq_len - 1)
    """

    def __init__(
        self,
        snippets:   List[CodeSnippet],
        config:     ModelConfig,
        tokenizer:  Optional[LanguageTokenizer] = None,
        stride:     Optional[int] = None,
        with_data_augmentation: bool = True,
        register_copies:        int = 3,
        seed:                   Optional[int] = None,
    ):
        self.config    = config
        self.tokenizer = tokenizer or LanguageTokenizer()
        self.stride    = stride or config.max_seq_len // 2

        # data augmentation
        self.register_copies  = register_copies
        self._rng             = random.Random(seed)

        self._sequences: List[torch.Tensor] = []
        for snippet in snippets:
            self._add(snippet.content, with_data_augmentation)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]      # (seq_len,)
        return seq[:-1], seq[1:]        # input, target; décalage d'un token

    # ── Construction ──────────────────────────────────────────────────────────

    def _add(self, source: str, with_data_augmentation: bool) -> None:
        # Original
        self._add_snippet(source)

        
        if with_data_augmentation:
            # Variantes registres
            for _ in range(self.register_copies):
                self._add_snippet(augment_registers(source, self._rng))

            # # Variante comparaison inversée FIXME
            # self._add_snippet(augment_flip_comparison(source, self._rng))
    

    def _add_snippet(self, source: str) -> None:
        # [BOS, t1, ..., tn, EOS] ; EOS présent, on veut que le modèle
        # apprenne à le prédire en fin de programme
        ids = self.tokenizer.encode(source, add_special_tokens=True)

        seq = torch.tensor(ids, dtype=torch.long)
        max_len = self.config.max_seq_len

        if len(seq) <= max_len:
            self._sequences.append(seq)
        else:
            # Fenêtres glissantes
            start = 0
            while start + 1 < len(seq):
                end = min(start + max_len, len(seq))
                self._sequences.append(seq[start:end])
                if end == len(seq):
                    break
                start += self.stride