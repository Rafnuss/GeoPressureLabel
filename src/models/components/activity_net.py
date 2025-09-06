import torch
from torch import nn


class ActivityNetBinary(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,  # helps capture patterns forward/backward
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # single logit for BCE
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)  # (B,1)


if __name__ == "__main__":
    _ = ActivityNetBinary()
