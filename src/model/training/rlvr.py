import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional

from src.data.snippet import CodeSnippet, parse_snippet_file
from src.data.specification import (
    ProgramSpecification, build_specification,
    SpecParseError, _run_with_inputs,
)
from src.data.file import flush_snippets, validate_snippet
from src.model.config import ModelConfig
from src.model.inference import Inference
from src.model.transformer import NextTokenTransformer
from src.model.training.trainer import Trainer, TrainConfig, resolve_model
from src.model.training.instruct import train_instruct

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class RLVRConfig:
    n_cycles:          int   = 5

    # Génération par cycle
    samples_per_cycle: int   = 50
    max_attempts:      int   = 300
    max_new_tokens:    int   = 256
    temperature:       float = 1.0
    top_k:             Optional[int] = 10

    # Fine-tune par cycle
    finetune_epochs:   int   = 5
    finetune_lr:       float = 5e-5
    batch_size:        int   = 4
    n_examples:        int   = 2 # dans le spécification

    # Tolérance numérique pour la vérification spec
    spec_tol:          float = 1e-3

    # Stockage
    output_dir:        str   = ".private/rlvr/data"
    checkpoint_dir:    str   = ".private/rlvr/checkpoints"
    device:            str   = "cpu"


# ─── Vérification ─────────────────────────────────────────────────────────────

def _check_specification(
    source: str,
    spec:   ProgramSpecification,
    tol:    float = 1e-3,
) -> bool:
    """
    Vérifie que le code source respecte tous les exemples de la spec.
    Retourne True si tous les exemples passent.
    """
    if not spec.examples:
        return True   # pas d'exemples, on ne peut pas vérifier

    for ex in spec.examples:
        outputs = _run_with_inputs(source, ex.inputs)
        if outputs is None:
            return False
        if len(outputs) != len(ex.outputs):
            return False
        for got, expected in zip(outputs, ex.outputs):
            # Booléens : comparaison stricte
            if isinstance(expected, bool):
                if got != expected:
                    return False
            # Numériques : tolérance
            else:
                try:
                    if abs(float(got) - float(expected)) > tol:
                        return False
                except (TypeError, ValueError):
                    return False
    return True


# ─── Boucle RLVR ─────────────────────────────────────────────────────────────

def rlvr_instruct(
    base_model:         str | NextTokenTransformer,
    seed_snippet_files: List[str],
    model_config:       ModelConfig  = None,
    config:             RLVRConfig   = None,
    on_cycle_start:     Callable[[int], None]                            | None = None,
    on_collected:       Callable[[int, int, int], None]                  | None = None,
    on_cycle_end:       Callable[[int, float, Optional[float], str], None] | None = None,
) -> NextTokenTransformer:
    """
    Boucle RLVR pour modèle instruct.

    À chaque cycle :
      1. Charge les snippets seeds, reconstruit leurs specs
      2. Pour chaque snippet, génère N complétions depuis la spec
      3. Garde celles qui passent _check_spec (I/O correct)
      4. Fine-tune sur seed + snippets validés (LR frais, faible)

    on_cycle_start(cycle)
    on_collected(cycle, n_accepted, n_attempts)
    on_cycle_end(cycle, train_loss, val_loss, checkpoint_path)
    """
    model_config = model_config or ModelConfig()
    config       = config       or RLVRConfig()

    output_dir = Path(config.output_dir)
    ckpt_dir   = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_checkpoint = resolve_model(base_model, model_config, ckpt_dir)

    # Charge les seeds une fois
    seed_snippets: List[CodeSnippet] = []
    for path in seed_snippet_files:
        seed_snippets.extend(parse_snippet_file(path))

    # Pré-construit les spécifications des seeds (stable entre les cycles)
    seed_specs: List[tuple[ProgramSpecification, CodeSnippet]] = []
    for snippet in seed_snippets:
        try:
            spec = build_specification(snippet, n_examples=config.n_examples)
            seed_specs.append((spec, snippet))
        except SpecParseError:
            pass

    if not seed_specs:
        raise ValueError("Aucun snippet seed avec signature valide.")

    # Replay buffer cumulatif
    generated_files: List[str] = []

    for cycle in range(1, config.n_cycles + 1):

        if on_cycle_start:
            on_cycle_start(cycle)

        # ── 1. Générer et filtrer ──────────────────────────────────────────────
        inference = Inference.from_checkpoint(current_checkpoint, device=config.device)
        accepted:  List[CodeSnippet] = []
        attempts   = 0

        while len(accepted) < config.samples_per_cycle and attempts < config.max_attempts:
            spec, seed = random.choice(seed_specs)
            attempts += 1

            tokens = inference.complete_instruct(
                specification  = spec,
                max_new_tokens = config.max_new_tokens,
                temperature    = config.temperature,
                top_k          = config.top_k,
            )
            generated = inference.tokenizer.tokens_to_source(tokens)

            candidate = CodeSnippet(
                name        = seed.name,
                description = seed.description,
                signature   = seed.signature,
                content     = generated,
            )

            # 1. Syntaxe et exécution valides (rejette les tokens de spec dans le code)
            ok, _ = validate_snippet(candidate)
            if not ok:
                continue

            # 2. Respect de la spec I/O
            if not _check_specification(generated, spec, tol=config.spec_tol):
                continue

            accepted.append(candidate)
        
        
        if on_collected:
            on_collected(cycle, len(accepted), attempts)

        # Flush des snippets validés
        cycle_file = output_dir / f"rlvr_cycle_{cycle:03d}"
        if accepted:
            flush_snippets(accepted, cycle_file)
            generated_files.append(str(cycle_file))

        # ── 2. Fine-tune — seed + replay buffer ───────────────────────────────
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
        
        train_instruct(
            snippet_files = seed_snippet_files + generated_files,
            model_config  = model_config,
            train_config  = cycle_train_config,
            on_epoch_end  = _on_epoch,
            augment       = False,
            n_examples    = config.n_examples,
        )

        raw_ckpt   = ckpt_dir / f"epoch_{config.finetune_epochs:04d}.pt"
        named_ckpt = ckpt_dir / f"rlvr_cycle_{cycle:03d}.pt"
        raw_ckpt.rename(named_ckpt)
        current_checkpoint = str(named_ckpt)

        if on_cycle_end:
            on_cycle_end(cycle, last_train_loss, last_val_loss, current_checkpoint)

    return Inference.from_checkpoint(current_checkpoint, device=config.device).model