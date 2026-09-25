import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PolyCLModel(nn.Module):
    def __init__(self, model_name: str, projection_dim: int = 128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        h = output.last_hidden_state[:, 0, :]
        z = F.normalize(self.projector(h), p=2, dim=-1)
        return h, z
