from dataclasses import dataclass
from src.tokenizer.vocabulary import VOCAB_SIZE, BOS_ID, EOS_ID, PAD_ID


@dataclass
class ModelConfig:
    # Vocab
    vocab_size:    int = VOCAB_SIZE

    # Architecture
    d_model:       int = 128
    n_heads:       int = 4
    n_layers:      int = 3
    d_feedforward: int = 512
    dropout:       float = 0.1

    # Séquences
    max_seq_len:   int = 512

    # Tokens spéciaux
    bos_id:        int = BOS_ID
    eos_id:        int = EOS_ID
    pad_id:        int = PAD_ID