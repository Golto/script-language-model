import re
import random
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from src.tokenizer import LanguageTokenizer

from src.model.config import ModelConfig


# ─── Augmentation ─────────────────────────────────────────────────────────────

_ALL_REGISTERS = [f"r{i}" for i in range(16)]

_REG_RE = re.compile(r'\br(\d+)\b')

_COMPARISON_FLIP = {
    '>': '<', 
    '<': '>', 
    '>=': '<=', 
    '<=': '>=',
    '==': '==',
    '!=': '!=',
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
    
    ---
    **Exemple** : source utilise r0, r1, r2
    1. tirage sans remise de 3 registres parmi les 16 (ex: r7, r3, r11)
    2. r0→r7, r1→r3, r2→r11 partout dans le source
    
    ---
    Garantit l'absence de conflits : le mapping est construit en une passe
    puis appliqué via un placeholder intermédiaire.
    """
    used = _used_registers(source)

    if not used:
        return source

    candidates = [r for r in _ALL_REGISTERS if r not in used]
    if len(candidates) < len(used):
        return source

    targets = rng.sample(candidates, k=len(used))
    mapping = dict(zip(used, targets))

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
    
    # Sous-pattern pour matcher un opérande valide
    # Registre (r1) OU Nombre flottant/entier (-0.5, 42) OU Booléen (true, false)
    operand = r'(r\d+|-?\d+(?:\.\d+)?|true|false)'
    
    # Pattern complet : opérande + opérateur + opérande
    pattern_str = fr'{operand}\s*(>=|<=|>|<|==|!=)\s*{operand}'
    pattern = re.compile(pattern_str)

    matches = list(pattern.finditer(source))
    if not matches:
        return source

    m = rng.choice(matches)
    left, op, right = m.group(1), m.group(2), m.group(3)
    
    flipped = f"{right} {_COMPARISON_FLIP[op]} {left}"
    return source[:m.start()] + flipped + source[m.end():]


# ─── BaseDataset ──────────────────────────────────────────────────────────────

class BaseDataset(Dataset):
    """
    Classe de base pour les datasets de séquences.
    Gère le sliding window et l'augmentation.
    Les sous-classes implémentent _encode(source) → List[int].
    """

    def __init__(
        self,
        config:           ModelConfig,
        tokenizer:        Optional[LanguageTokenizer] = None,
        stride:           Optional[int] = None,
        augment:          bool = True,
        register_copies:  int  = 3,
        seed:             Optional[int] = None,
    ):
        self.config          = config
        self.tokenizer       = tokenizer or LanguageTokenizer()
        self.stride          = stride or config.max_seq_len // 2

        # Data Augmentation
        self.augment         = augment
        self.register_copies = register_copies
        self._rng            = random.Random(seed)

        self._sequences: List[torch.Tensor] = []


    # ── Indexing ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]      # (seq_len,)
        return seq[:-1], seq[1:]        # input, target; décalage d'un token


    # ── Interface sous-classes ────────────────────────────────────────────────

    def _encode(self, source: str) -> List[int]:
        """Retourne la séquence d'ids complète (BOS inclus, EOS inclus)."""
        raise NotImplementedError

    def _augment_source(self, source: str) -> str:
        for _ in range(self.register_copies):
            variant = augment_registers(source, self._rng)
            self._push(self._encode(variant))
        
        variant = augment_flip_comparison(source, self._rng)
        self._push(self._encode(variant))


    # ── Ajout ─────────────────────────────────────────────────────────────────

    def add(self, source: str) -> None:
        """Encode et ajoute le source (+ variantes augmentées)."""
        self._push(self._encode(source))
        if self.augment:
            self._augment_source(source)
    
    def _push(self, ids: List[int]) -> None:
        """Découpe en fenêtres glissantes et ajoute dans _sequences."""
        seq = torch.tensor(ids, dtype=torch.long)
        max_len = self.config.max_seq_len
        
        if len(seq) <= max_len:
            self._sequences.append(seq)
        else:
            start = 0
            while start + 1 < len(seq):
                end = min(start + max_len, len(seq))
                self._sequences.append(seq[start:end])
                if end == len(seq):
                    break
                start += self.stride