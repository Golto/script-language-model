from typing import List, Callable

from src.model.config import ModelConfig
from src.model.datasets.foundation import FoundationDataset
from src.model.transformer import NextTokenTransformer

from .trainer import Trainer, TrainConfig


def train_foundation(
    snippet_files:        List[str],
    model_config:         ModelConfig          = None,
    train_config:         TrainConfig          = None,
    on_epoch_end:         Callable | None      = None,
    augment:              bool                 = True,
) -> NextTokenTransformer:

    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()

    dataset = FoundationDataset.from_files(
        snippet_files,
        config = model_config,
        augment = augment,
    )

    return Trainer(model_config, train_config).run(dataset, on_epoch_end)