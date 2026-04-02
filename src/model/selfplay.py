from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from src.data.snippet import CodeSnippet
from src.data.file import validate_snippet, flush_snippets
from src.model.config import ModelConfig
from src.model.inference import Inference
from src.model.train import TrainConfig, train
from src.model.transformer import NextTokenTransformer


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
    top_k:              int | None = 10

    # Fine-tuning par cycle; LR réintialisé, peu d'epochs
    finetune_epochs:    int   = 10
    finetune_lr:        float = 1e-4
    batch_size:         int   = 4

    # Contraintes sur les snippets générés
    # TODO généraliser à un scoring
    require_output:     bool  = True
    require_input:      bool  = False

    # Stockage
    output_dir:         str   = ".private/selfplay/data"
    checkpoint_dir:     str   = ".private/selfplay/checkpoints"
    device:             str   = "cpu"


# ─── Génération filtrée ───────────────────────────────────────────────────────

def _collect_valid_snippets(
    inference:      Inference,
    config:         SelfPlayConfig,
    on_progress:    Callable[[int, int], None] | None = None,
) -> List[CodeSnippet]:
    """
    Génère des complétions depuis des prompts aléatoires
    et ne garde que les snippets valides selon les contraintes.
    """
    valid:   List[CodeSnippet] = []
    prompts  = config.prompts or ["input r0\n", "input r0\ninput r1\n", ""]

    for attempt in range(1, config.max_attempts + 1):
        if len(valid) >= config.samples_per_cycle:
            break

        prompt = random.choice(prompts)
        response = inference.complete(
            prompt          = prompt,
            max_new_tokens  = config.max_new_tokens,
            temperature     = config.temperature,
            top_k           = config.top_k,
        )

        snippet = CodeSnippet(
            name        = "selfplay",
            description = "auto-generated snippet",
            signature   = "[unknown] -> unknown",
            content     = response,
        )

        ok, _ = validate_snippet(snippet)
        if not ok:
            continue
        if config.require_output and "output" not in snippet.content:
            continue
        if config.require_input and "input" not in snippet.content:
            continue

        valid.append(snippet)
        if on_progress:
            on_progress(len(valid), attempt)

    return valid


# ─── Boucle self-play ─────────────────────────────────────────────────────────

def self_play(
    base_checkpoint:    str,
    seed_snippet_files: List[str],
    model_config:       ModelConfig     = None,
    config:             SelfPlayConfig  = None,
    on_cycle_start:     Callable[[int], None]                            | None = None,
    on_collected:       Callable[[int, List[CodeSnippet]], None]         | None = None,
    on_cycle_end:       Callable[[int, float, float | None, str], None]  | None = None,
) -> NextTokenTransformer:
    """
    Boucle self-play itérative.

    À chaque cycle :
      1. Génère des snippets valides avec le modèle courant
      2. Fine-tune sur seed + snippets générés (LR frais)
      3. Sauvegarde le checkpoint du cycle

    on_cycle_end(cycle, train_loss, val_loss, checkpoint_path)

    Retourne le modèle du dernier cycle.
    """
    model_config = model_config or ModelConfig()
    config       = config       or SelfPlayConfig()

    output_dir  = Path(config.output_dir)
    ckpt_dir    = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint = base_checkpoint

    for cycle in range(1, config.n_cycles + 1):

        if on_cycle_start:
            on_cycle_start(cycle)

        # ── 1. Générer des snippets valides ───────────────────────────────────
        inference = Inference.from_checkpoint(current_checkpoint, device=config.device)

        def _progress(n_valid, n_attempts):
            print(f"\r  collecte : {n_valid:3d} / {config.samples_per_cycle}"
                  f"  (tentatives : {n_attempts})", end="")

        snippets = _collect_valid_snippets(inference, config, on_progress=_progress)
        print()  # newline après le \r

        if on_collected:
            on_collected(cycle, snippets)

        # Flush les snippets générés dans un fichier dédié au cycle
        cycle_file = output_dir / f"cycle_{cycle:03d}"
        flush_snippets(snippets, cycle_file)

        # ── 2. Fine-tune sur seed + snippets du cycle ─────────────────────────
        # LR réinitialisé : on construit un TrainConfig sans resume_from sur l'optimizer
        # mais on repart des POIDS du cycle précédent via resume_from
        cycle_train_config = TrainConfig(
            snippet_files = seed_snippet_files + [str(cycle_file)],
            epochs        = config.finetune_epochs,
            batch_size    = config.batch_size,
            lr            = config.finetune_lr,
            checkpoint_dir= str(ckpt_dir),
            save_every    = config.finetune_epochs,  # save uniquement à la fin
            device        = config.device,
            resume_from   = current_checkpoint,      # poids uniquement
        )

        last_train_loss = None
        last_val_loss   = None

        def _on_epoch(epoch, train_loss, val_loss):
            nonlocal last_train_loss, last_val_loss
            last_train_loss = train_loss
            last_val_loss   = val_loss
            print(f"  epoch {epoch:3d} | train {train_loss:.4f}"
                  + (f" | val {val_loss:.4f}" if val_loss else ""))

        train(
            model_config  = model_config,
            train_config  = cycle_train_config,
            on_epoch_end  = _on_epoch,
            with_data_augmentation = True,
            reset_optimizer = True,
        )

        cycle_ckpt = str(ckpt_dir / f"epoch_{config.finetune_epochs:04d}.pt")
        named_ckpt = str(ckpt_dir / f"cycle_{cycle:03d}.pt")
        Path(cycle_ckpt).rename(named_ckpt)
        current_checkpoint = named_ckpt

        if on_cycle_end:
            on_cycle_end(cycle, last_train_loss, last_val_loss, named_ckpt)

    return Inference.from_checkpoint(current_checkpoint, device=config.device).model