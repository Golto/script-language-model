from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

from src.model.config import ModelConfig
from src.model.transformer import NextTokenTransformer
from src.tokenizer import LanguageTokenizer
from src.tokenizer.vocabulary import BOS_ID


# ─── Chargement ───────────────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[NextTokenTransformer, ModelConfig, dict]:
    """
    Charge un checkpoint et retourne (model, config, meta).
    meta contient epoch, train_loss, val_loss.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    config = ckpt.get("model_config") or ModelConfig()
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

    # ── API publique ──────────────────────────────────────────────────────────

    def complete(
        self,
        prompt:         str,
        max_new_tokens: int   = 128,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
    ) -> str:
        """
        Complète un prompt source et retourne le programme généré (prompt inclus).
        Le prompt est encodé sans BOS/EOS — on ajoute BOS manuellement
        pour que le modèle soit dans les mêmes conditions qu'à l'entraînement,
        sans EOS prématuré qui tronquerait la génération.
        """
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = [BOS_ID] + ids

        prompt_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

        output_ids = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # (1, seq_len) → List[int], on retire le BOS initial
        flat = output_ids[0].tolist()
        return self.tokenizer.decode_str(flat[1:])  # skip BOS, garde EOS filtré par decode_str

    def info(self) -> str:
        """Résumé du checkpoint chargé."""
        lines = [
            f"Modèle    : {self.model}",
            f"Epoch     : {self.meta.get('epoch', '?')}",
            f"Train loss: {self.meta.get('train_loss', '?'):.4f}" if self.meta.get('train_loss') else "Train loss: ?",
            f"Val loss  : {self.meta.get('val_loss', '?'):.4f}"   if self.meta.get('val_loss')   else "Val loss  : ?",
        ]
        return "\n".join(lines)