from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import torch

from src.data.specification import ProgramSpecification
from src.model.config import ModelConfig
from src.model.transformer import NextTokenTransformer
from src.tokenizer import LanguageTokenizer
from src.tokenizer.vocabulary import BOS_ID


# ─── Chargement ───────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Tuple[NextTokenTransformer, ModelConfig, dict]:
    """
    Charge un checkpoint et retourne (model, config, meta).
    meta contient epoch, train_loss, val_loss.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)

    config_dict = ckpt.get("model_config")
    config = ModelConfig(**config_dict) if isinstance(config_dict, dict) else ModelConfig()

    model  = NextTokenTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "epoch":      ckpt.get("epoch"),
        "train_loss": ckpt.get("train_loss"),
        "val_loss":   ckpt.get("val_loss"),
    }

    return model, config, meta


# ─── Inférence ────────────────────────────────────────────────────────────────

class Inference:
    """
    Interface de test d'un modèle chargé depuis un checkpoint.

    Exemple :
        inf = Inference.from_checkpoint("checkpoints/epoch_0050.pt")
        print(inf.complete("r0 = 4\\n"))
        print(inf.complete("input r0\\n", temperature=0.8, top_k=10))
    """

    def __init__(
        self,
        model:     NextTokenTransformer,
        config:    ModelConfig,
        tokenizer: LanguageTokenizer | None = None,
        device:    str = "cpu",
        meta:      dict | None = None,
    ):
        self.model     = model
        self.config    = config
        self.tokenizer = tokenizer or LanguageTokenizer()
        self.device    = torch.device(device)
        self.meta      = meta or {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> "Inference":
        model, config, meta = load_checkpoint(checkpoint_path, device)
        return cls(model=model, config=config, device=device, meta=meta)
    
    
    # ── Génération ────────────────────────────────────────────────────────────

    def _generate(
        self,
        prompt_ids:             List[int],
        max_new_tokens:         int,
        temperature:            float,
        top_k:                  Optional[int],
        skip_special_tokens:    bool = True,
        indent:                 int = 4,
    ) -> str:
        """Génère depuis une séquence d'ids et retourne le texte décodé."""
        tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        output = self.model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        flat = output[0].tolist()
        return self.tokenizer.decode_str(
            flat[1:], # retirer BOS
            skip_special_tokens=skip_special_tokens,
            indent=indent,
        )

    # ── Modèle de fondation ───────────────────────────────────────────────────

    def complete(
        self,
        prompt:         str,
        max_new_tokens: int            = 128,
        temperature:    float          = 1.0,
        top_k:          Optional[int]  = None,
        indent:         int            = 4,
    ) -> str:
        """
        Complète un prompt source (modèle de fondation).
        """
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = [BOS_ID] + ids

        return self._generate(
            prompt_ids     = ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            indent         = indent,
        )
    
    
    # ── Modèle Instruct ───────────────────────────────────────────────────────

    def complete_instruct(
        self,
        specification:  ProgramSpecification,
        prompt:         str | None     = None,
        max_new_tokens: int            = 128,
        temperature:    float          = 1.0,
        top_k:          Optional[int]  = None,
        indent:         int            = 4,
    ) -> str:
        """
        Complète depuis une spécification (modèle instruct).

        La séquence de prompt est : BOS + spec_tokens [+ code_partiel]
        Le modèle génère le code qui suit la spécification.

        prompt : code partiel optionnel ajouté après la spécification
        """
        # BOS + spec (sans EOS, le modèle complète)
        ids = self.tokenizer.encode_instruct(specification, source=None)

        # code partiel optionnel ajouté après la spec
        if prompt:
            partial_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = ids + partial_ids

        return self._generate(
            prompt_ids     = ids,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            indent         = indent,
        )
    

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self) -> str:
        """Résumé du checkpoint chargé."""
        lines = [
            f"Modèle    : {self.model}",
            f"Epoch     : {self.meta.get('epoch', '?')}",
            f"Train loss: {self.meta.get('train_loss', '?'):.4f}" if self.meta.get('train_loss') else "Train loss: ?",
            f"Val loss  : {self.meta.get('val_loss', '?'):.4f}"   if self.meta.get('val_loss')   else "Val loss  : ?",
        ]
        return "\n".join(lines)