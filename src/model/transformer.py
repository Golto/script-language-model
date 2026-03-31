import torch
import torch.nn as nn
from .config import ModelConfig
from .embedding import TokenEmbedding


class NextTokenTransformer(nn.Module):
    """
    Transformer décodeur causal pour la prédiction du prochain token.

    Entrée  : token_ids  (batch, seq_len)
    Sortie  : logits     (batch, seq_len, vocab_size)

    Entraînement : CrossEntropyLoss(logits[:, :-1], token_ids[:, 1:])
    soit logits sur input[:-1] vs target = input[1:]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(config)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN : entraînement plus stable
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Partage de poids embedding ↔ projection finale (Press & Wolf, 2017)
        # Réduit les paramètres, améliore la généralisation sur petit vocab
        self.head.weight = self.embedding.embedding.weight

        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids : (batch, seq_len)
        seq_len = token_ids.size(1)
        device  = token_ids.device

        x = self.embedding(token_ids)             # (batch, seq_len, d_model)

        # Masque causal : position i ne voit que 0..i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        # TransformerDecoder sans mémoire encoder = décodeur causal pur
        # On passe x comme tgt ET memory avec un memory_mask bloquant tout
        out = self.decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )                                          # (batch, seq_len, d_model)

        return self.head(out)                      # (batch, seq_len, vocab_size)

    # ── Génération ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Génère des tokens à partir d'un prompt.
        prompt_ids : (1, seq_len) — batch size 1 uniquement
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Tronque si on dépasse max_seq_len
            ctx = ids[:, -self.config.max_seq_len:]
            logits = self(ctx)                     # (1, seq_len, vocab_size)
            next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            if top_k is not None:
                top_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            ids = torch.cat([ids, next_id], dim=1)

            if next_id.item() == self.config.eos_id:
                break

        return ids

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"NextTokenTransformer("
            f"vocab={self.config.vocab_size}, "
            f"d_model={self.config.d_model}, "
            f"layers={self.config.n_layers}, "
            f"heads={self.config.n_heads}, "
            f"params={self.num_parameters():,})"
        )