import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Pass the input through the LSTM layer
        lstm_out, _ = self.lstm(x)

        # Pass the output of the LSTM through the fully connected layer
        output = self.fc(lstm_out[:, -1, :])

        return output
