import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSpinal3DNetImproved(nn.Module):
    """
    Single disease, improved pooling. If binary => num_classes=1
    """
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1,2,2))

        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.AdaptiveAvgPool3d((4,12,12))

        self.dropout3d = nn.Dropout3d(dropout_prob)

        self.lstm = nn.LSTM(128*12*12, 256, 2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # shape still 5D => dropout3d is valid
        x = self.dropout3d(x)

        b, c, d, h, w = x.shape
        # Flatten => [B, D, c*h*w]
        x = x.view(b, d, -1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return self.fc(context)
