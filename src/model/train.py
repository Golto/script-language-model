import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Callable

from src.data.snippet import parse_snippet_file, CodeSnippet
from src.tokenizer import LanguageTokenizer
from .config import ModelConfig
from .dataset import ProgramDataset
from .transformer import NextTokenTransformer



# ─── Config d'entraînement ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Données
    snippet_files:  List[str]   = field(default_factory=list)
    val_split:      float       = 0.1

    # Entraînement
    epochs:         int         = 50
    batch_size:     int         = 4
    lr:             float       = 3e-4
    weight_decay:   float       = 1e-2
    grad_clip:      float       = 1.0

    # Reprise
    resume_from:    str | None  = None 

    # Sauvegarde
    checkpoint_dir: str         = ".private/checkpoints"
    save_every:     int         = 10 # chaque N epochs

    # Device
    device:         str         = "cpu"


# ─── Collate ──────────────────────────────────────────────────────────────────

def _collate(batch, pad_id: int):
    """Pad les séquences à la longueur max du batch."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    def pad(seqs):
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.size(0),), pad_id, dtype=torch.long)])
            for s in seqs
        ])

    return pad(inputs), pad(targets)


# ─── Boucle d'entraînement ────────────────────────────────────────────────────

def train(
    model_config:   ModelConfig  = None,
    train_config:   TrainConfig  = None,
    on_epoch_end:   Callable[[int, float, float | None], None] | None = None,
    with_data_augmentation: bool = True,
) -> NextTokenTransformer:
    """
    Entraîne le modèle et retourne le modèle entraîné.

    on_epoch_end(epoch, train_loss, val_loss | None) — callback optionnel
    pour logger / afficher la progression.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    device       = torch.device(train_config.device)

    # ── Données ───────────────────────────────────────────────────────────────
    snippets: List[CodeSnippet] = []
    for path in train_config.snippet_files:
        snippets.extend(parse_snippet_file(path))

    if not snippets:
        raise ValueError("Aucun snippet chargé — vérifier snippet_files dans TrainConfig")

    tokenizer = LanguageTokenizer()
    dataset   = ProgramDataset(snippets, model_config, tokenizer, with_data_augmentation=with_data_augmentation)

    val_size   = max(1, int(len(dataset) * train_config.val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    pad_id = model_config.pad_id
    collate = lambda b: _collate(b, pad_id)

    train_loader = DataLoader(train_set, batch_size=train_config.batch_size,
                              shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_set,   batch_size=train_config.batch_size,
                              shuffle=False, collate_fn=collate)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model     = NextTokenTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_config.lr,
                                  weight_decay=train_config.weight_decay)

    # ── Reprise depuis checkpoint ─────────────────────────────────────────────
    start_epoch = 1
    if train_config.resume_from:
        ckpt = torch.load(train_config.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis epoch {ckpt['epoch']} "
              f"(train={ckpt.get('train_loss', '?'):.4f})")

    # epochs / remaining epochs
    remaining_epochs = train_config.epochs - (start_epoch - 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(remaining_epochs, 1)
    )

    # NOTE <EOS> 5x poids
    weight = torch.ones(model_config.vocab_size)
    weight[model_config.eos_id] = 5.0   
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, weight=weight.to(device))

    Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Epochs ────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, train_config.epochs + 1):

        # — Train —
        model.train()
        train_loss = 0.0
        for input_ids, target_ids in train_loader:
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            # logits : (batch, seq_len, vocab_size)
            # target : (batch, seq_len)
            loss = criterion(
                logits.reshape(-1, model_config.vocab_size),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # — Validation —
        val_loss = None
        if val_set:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input_ids, target_ids in val_loader:
                    input_ids  = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    logits     = model(input_ids)
                    val_loss  += criterion(
                        logits.reshape(-1, model_config.vocab_size),
                        target_ids.reshape(-1),
                    ).item()
            val_loss /= len(val_loader)

        # — Callback —
        if on_epoch_end:
            on_epoch_end(epoch, train_loss, val_loss)

        # — Checkpoint —
        if epoch % train_config.save_every == 0:
            ckpt_path = Path(train_config.checkpoint_dir) / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "train_loss":   train_loss,
                "val_loss":     val_loss,
                "model_config": asdict(model_config),
            }, ckpt_path)

    return model