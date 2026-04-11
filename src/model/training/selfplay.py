from __future__ import annotations

import random
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional



from src.data.snippet import CodeSnippet
from src.data.file import validate_snippet, flush_snippets
from src.data.scorer import SnippetScorer
from src.model.config import ModelConfig
from src.model.inference import Inference
from src.model.transformer import NextTokenTransformer

from .trainer import TrainConfig
from .foundation import train_foundation


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class SelfPlayConfig:
    # Cycles
    n_cycles:           int   = 5

    # Génération par cycle
    prompts:            List[str] = field(default_factory=list)
    samples_per_cycle:  int   = 100
    max_attempts:       int   = 500
    max_new_tokens:     int   = 64
    temperature:        float = 1.0
    top_k:              Optional[int] = 20

    # Fine-tuning par cycle; LR réintialisé, peu d'epochs
    finetune_epochs:    int   = 3
    finetune_lr:        float = 1e-5
    batch_size:         int   = 16
    augment:            bool  = True

    # Stockage
    output_dir:         str   = ".private/selfplay/data"
    checkpoint_dir:     str   = ".private/selfplay/checkpoints"
    device:             str   = "cpu"


# ─── Scorer par défaut ────────────────────────────────────────────────────────

def default_scorer(snippet: CodeSnippet) -> bool:
    """Filtre minimal : exécution valide + présence d'un output."""
    ok, _ = validate_snippet(snippet)
    return ok and "output" in snippet.content


# ─── Génération filtrée ───────────────────────────────────────────────────────

def collect_valid_snippets(
    inference:   Inference,
    config:      SelfPlayConfig,
    scorer:      SnippetScorer = default_scorer,
    on_progress: Callable[[int, int], None] | None = None,
) -> List[CodeSnippet]:
    """
    Génère des complétions depuis des prompts aléatoires
    et ne garde que les snippets acceptés par le scorer.
    """
    valid:   List[CodeSnippet] = []
    prompts  = config.prompts or ["input r0\n", "input r0\ninput r1\n", ""]

    for attempt in range(1, config.max_attempts + 1):
        if len(valid) >= config.samples_per_cycle:
            break

        prompt = random.choice(prompts)
        tokens = inference.complete(
            prompt          = prompt,
            max_new_tokens  = config.max_new_tokens,
            temperature     = config.temperature,
            top_k           = config.top_k,
        )
        response = inference.tokenizer.tokens_to_source(tokens)

        snippet = CodeSnippet(
            name        = "selfplay",
            description = "auto-generated snippet",
            signature   = "[unknown] -> unknown",
            content     = response,
        )

        if not scorer(snippet):
            continue

        valid.append(snippet)
        if on_progress:
            on_progress(len(valid), attempt)

    return valid


# ─── Helpers checkpoint ───────────────────────────────────────────────────────

def _save_temp_checkpoint(
    model: NextTokenTransformer,
    model_config: ModelConfig,
    path: Path,
) -> None:
    """Sauvegarde un modèle en mémoire dans un fichier temporaire."""
    from dataclasses import asdict
    torch.save({
        "epoch":        0,
        "model_state":  model.state_dict(),
        "optimizer":    {},
        "train_loss":   None,
        "val_loss":     None,
        "model_config": asdict(model_config),
    }, path)


def _resolve_model(
    base_model: str | NextTokenTransformer,
    model_config: ModelConfig,
    checkpoint_dir: Path,
) -> str:
    """
    Retourne un chemin de checkpoint utilisable.
    Si base_model est déjà un chemin → retourné tel quel.
    Si base_model est un NextTokenTransformer → sauvegardé dans un .pt temporaire.
    """
    if isinstance(base_model, str):
        return base_model

    tmp_path = checkpoint_dir / "base_model_tmp.pt"
    _save_temp_checkpoint(base_model, model_config, tmp_path)
    return str(tmp_path)


# ─── Boucle self-play ─────────────────────────────────────────────────────────

def self_play(
    base_model:         str | NextTokenTransformer,
    seed_snippet_files: List[str],
    model_config:       ModelConfig    = None,
    config:             SelfPlayConfig = None,
    scorer:             SnippetScorer  = default_scorer,
    on_cycle_start:     Callable[[int], None]                           | None = None,
    on_collected:       Callable[[int, List[CodeSnippet]], None]        | None = None,
    on_cycle_end:       Callable[[int, float, Optional[float], str], None] | None = None,
) -> NextTokenTransformer:
    """
    Boucle self-play itérative.

    À chaque cycle :
      1. Génère des snippets valides avec le modèle courant (filtrés par scorer)
      2. Fine-tune sur seed + snippets générés (LR frais, poids du cycle précédent)
      3. Sauvegarde le checkpoint du cycle

    base_model : chemin vers un .pt existant, OU un NextTokenTransformer en mémoire
                 (sauvegardé temporairement, le fichier original n'est pas écrasé)

    on_cycle_end(cycle, train_loss, val_loss, checkpoint_path)
    """
    model_config = model_config or ModelConfig()
    config       = config       or SelfPlayConfig()

    output_dir = Path(config.output_dir)
    ckpt_dir   = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint = _resolve_model(base_model, model_config, ckpt_dir)

    # replay buffer (accumule plusieurs fichiers)
    generated_files: List[str] = []

    for cycle in range(1, config.n_cycles + 1):

        if on_cycle_start:
            on_cycle_start(cycle)

        # ── 1. Collecter des snippets valides ─────────────────────────────────
        inference = Inference.from_checkpoint(current_checkpoint, device=config.device)

        def _progress(n_valid, n_attempts):
            print(f"\r  collecte : {n_valid:3d} / {config.samples_per_cycle}"
                  f"  (tentatives : {n_attempts})", end="")

        snippets = collect_valid_snippets(inference, config, scorer, _progress)
        print()

        if on_collected:
            on_collected(cycle, snippets)

        cycle_file = output_dir / f"cycle_{cycle:03d}"
        flush_snippets(snippets, cycle_file)
        generated_files.append(str(cycle_file))

        # ── 2. Fine-tune — poids du cycle précédent, LR frais ─────────────────
        named_ckpt = ckpt_dir / f"cycle_{cycle:03d}.pt"

        cycle_train_config = TrainConfig(
            epochs          = config.finetune_epochs,
            batch_size      = config.batch_size,
            lr              = config.finetune_lr,
            checkpoint_dir  = str(ckpt_dir),
            save_every      = config.finetune_epochs,
            device          = config.device,
            resume_from     = current_checkpoint,
            reset_optimizer = True,
        )

        last_train_loss: Optional[float] = None
        last_val_loss:   Optional[float] = None

        def _on_epoch(epoch, train_loss, val_loss):
            nonlocal last_train_loss, last_val_loss
            last_train_loss = train_loss
            last_val_loss   = val_loss
            print(f"  epoch {epoch:3d} | train {train_loss:.4f}"
                  + (f" | val {val_loss:.4f}" if val_loss else ""))

        train_foundation(
            snippet_files = seed_snippet_files + generated_files,
            model_config  = model_config,
            train_config  = cycle_train_config,
            on_epoch_end  = _on_epoch,
            augment       = config.augment,
        )

        # Renomme l'epoch_XXXX.pt → cycle_XXX.pt pour tracer les cycles
        raw_ckpt   = ckpt_dir / f"epoch_{config.finetune_epochs:04d}.pt"
        named_ckpt = ckpt_dir / f"cycle_{cycle:03d}.pt"
        raw_ckpt.rename(named_ckpt)
        current_checkpoint = str(named_ckpt)

        if on_cycle_end:
            on_cycle_end(cycle, last_train_loss, last_val_loss, current_checkpoint)

    return Inference.from_checkpoint(current_checkpoint, device=config.device).model



