from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable, Optional

from src.model.config import ModelConfig
from src.model.transformer import NextTokenTransformer


# ─── Config d'entraînement ────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Entraînement
    epochs:             int   = 50
    batch_size:         int   = 4
    lr:                 float = 3e-4
    weight_decay:       float = 1e-2
    grad_clip:          float = 1.0
    val_split:          float = 0.1

    # Reprise
    resume_from:        Optional[str] = None
    reset_optimizer:    bool          = False # lr réinitialisé si True

    # Sauvegarde
    checkpoint_dir:     str = ".private/checkpoints"
    save_every:         int = 10 # epochs

    # Device
    device:             str = "cpu"


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


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Boucle d'entraînement commune pour foundation et instruct.
    Reçoit un dataset déjà construit et gère :
      - split train/val
      - DataLoader
      - optimizer + scheduler + criterion
      - reprise depuis checkpoint (poids seuls ou poids + optimizer)
      - sauvegarde périodique
      - callback on_epoch_end
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device       = torch.device(train_config.device)

        self.generator = torch.Generator().manual_seed(42) # Fix: évite data leak


    def run(
        self,
        dataset:      Dataset,
        on_epoch_end: Callable[[int, float, Optional[float]], None] | None = None,
    ) -> NextTokenTransformer:
        
        cfg = self.train_config
        pad_id = self.model_config.pad_id

        # ── Split ─────────────────────────────────────────────────────────────
        val_size   = max(1, int(len(dataset) * cfg.val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=self.generator)

        collate = lambda batch: _collate(batch, pad_id)
        train_loader = DataLoader(
            train_set, 
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate
        )
        val_loader = DataLoader(
            val_set,   
            batch_size=cfg.batch_size,
            shuffle=False, 
            collate_fn=collate
        )

        # ── Modèle & optimizer ────────────────────────────────────────────────
        model     = NextTokenTransformer(self.model_config).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # ── Reprise ───────────────────────────────────────────────────────────
        
        start_epoch = 1
        
        if cfg.resume_from:
            ckpt = torch.load(cfg.resume_from, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            
            if not cfg.reset_optimizer:
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt["epoch"] + 1

                # Fix: lr = 0.0 de l'entraînement précédent écrasé par lr initial (pour CosineAnnealing)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr
                    param_group['initial_lr'] = cfg.lr
            else:
                start_epoch = 1
                print(f"Reprise depuis epoch {ckpt['epoch']} (Optimizer réinitialisé)")

        # ── Scheduler ─────────────────────────────────────────────────────────
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        # Fix: recalculer/reprendre le lr de la bonne epoch
        if cfg.resume_from and not cfg.reset_optimizer:
            
            old_t_max = ckpt.get("scheduler", {}).get("T_max", cfg.epochs)
            
            if "scheduler" in ckpt and old_t_max == cfg.epochs:
                # si epochs totales inchangées
                scheduler.load_state_dict(ckpt["scheduler"])
                print("Scheduler chargé depuis le checkpoint.")
                
            else:
                # si prolongation de l'entraînement ou scheduler pas sauvegardé
                print(f"Recalcul du lr pour {cfg.epochs} epochs totales...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr
                    param_group['initial_lr'] = cfg.lr
                    
                for _ in range(ckpt["epoch"]):
                    scheduler.step()
                    
            print(f"Reprise depuis epoch {ckpt['epoch']} | lr : {optimizer.param_groups[0]['lr']:.2e}")


        # ── Boucle epochs ─────────────────────────────────────────────────────
        
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, cfg.epochs + 1):

            train_loss = self._train_epoch(
                model, 
                train_loader, 
                optimizer,
                criterion, 
                self.model_config.vocab_size
            )
            scheduler.step()

            val_loss = self._val_epoch(
                model, 
                val_loader, 
                criterion,
                self.model_config.vocab_size
            )

            if on_epoch_end:
                on_epoch_end(epoch, train_loss, val_loss)

            if epoch % cfg.save_every == 0:
                self._save(model, optimizer, scheduler, epoch, train_loss, val_loss)

        return model

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(
        self, 
        model: NextTokenTransformer, 
        loader: DataLoader, 
        optimizer: Optimizer, 
        criterion: nn.CrossEntropyLoss, 
        vocab_size: int
    ) -> float:
        model.train()
        total = 0.0

        for input_ids, target_ids in loader:
            input_ids  = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            logits = model(input_ids)

            # logits : (batch, seq_len, vocab_size)
            # target : (batch, seq_len)
            loss = criterion(
                logits.reshape(-1, vocab_size), 
                target_ids.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.train_config.grad_clip)
            optimizer.step()

            total += loss.item()

        return total / len(loader)
    

    def _val_epoch(
        self, 
        model: NextTokenTransformer, 
        loader: DataLoader, 
        criterion: nn.CrossEntropyLoss, 
        vocab_size: int
    ) -> Optional[float]:
        
        if not loader:
            return None
        
        model.eval()
        total = 0.0
        with torch.no_grad():
            for input_ids, target_ids in loader:
                input_ids  = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                logits     = model(input_ids)
                total += criterion(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1)
                ).item()
        return total / len(loader)
    

    def _save(
        self, 
        model: NextTokenTransformer, 
        optimizer: Optimizer, 
        scheduler: LRScheduler,
        epoch: int, 
        train_loss: float, 
        val_loss: float
    ):
        path = Path(self.train_config.checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "model_config": asdict(self.model_config),
        }, path)


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


def resolve_model(
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