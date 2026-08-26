import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = None
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        b, c, f, t = x.size()

        x = x.view(b, 1, c*f, t)
        x = self.cnn(x)

        b, c, f, t = x.size()
        x = x.view(b, t, c*f)

        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=c*f,
                hidden_size=128,
                batch_first=True,
                bidirectional=True
            ).to(x.device)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return self.fc(x)