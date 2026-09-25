from pathlib import Path

import torch
import torch.nn as nn

from model import PolyCLModel


class TgRegressor(nn.Module):
    def __init__(self, model_name, checkpoint_path):
        super().__init__()

        pretrained = PolyCLModel(model_name)
        state = torch.load(
            Path(checkpoint_path),
            map_location="cpu",
        )
        pretrained.load_state_dict(state)

        self.encoder = pretrained.encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.head = nn.Linear(
            self.encoder.config.hidden_size,
            1,
        )

    def forward(self, input_ids, attention_mask):
        self.encoder.eval()

        with torch.no_grad():
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            h = output.last_hidden_state[:, 0, :]

        return self.head(h).squeeze(-1)
