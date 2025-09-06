import torch
from torch import nn


class ActivityNetBinaryCNN(nn.Module):
    def __init__(self, input_dim, num_filters=64, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_dim,
                num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                num_filters,
                num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, 1),  # single logit for BCE
        )

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = x.transpose(1, 2)  # (B, input_dim, seq_len) for Conv1d
        out = self.conv(x)  # (B, num_filters, seq_len)
        out = out.mean(dim=2)  # Global average pooling over time
        return self.fc(out)  # (B,1)


if __name__ == "__main__":
    _ = ActivityNetBinaryCNN(input_dim=2)
