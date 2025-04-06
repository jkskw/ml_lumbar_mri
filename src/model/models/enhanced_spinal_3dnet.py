import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSpinal3DNet(nn.Module):
    """
    EnhancedSpinal3DNet is an improved version of your best model.
    
    Changes compared to AdvancedSpinal3DNetImproved:
      • After the 3D conv blocks and adaptive pooling, a 1×1×1 conv (with BN and ReLU)
        reduces the channel dimension from 128 to 64.
      • This lowers the per–slice feature size (from 128*12*12 to 64*12*12), making the LSTM’s
        job easier and reducing over–parameterization.
      • A two–layer bidirectional LSTM with dropout and an attention mechanism is used to
        aggregate the slice features.
    """
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super().__init__()
        # 3D Convolutional Backbone
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1,2,2))  # Keep depth
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        # Pool to fixed dimensions: depth=4, height=12, width=12
        self.pool3 = nn.AdaptiveAvgPool3d((4,12,12))
        
        # 1x1x1 convolution to reduce channel dimension from 128 to 64
        self.conv_reduce = nn.Conv3d(128, 64, kernel_size=1)
        self.bn_reduce   = nn.BatchNorm3d(64)
        
        self.dropout3d = nn.Dropout3d(dropout_prob)
        
        # LSTM Aggregator
        # After reduction, each slice has feature vector of size 64*12*12 = 9216
        self.lstm = nn.LSTM(64 * 12 * 12, 256, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout_prob)
        self.dropout_lstm = nn.Dropout(dropout_prob)
        
        # Attention mechanism to aggregate LSTM outputs (which are 512–dimensional per slice)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: [B, 1, D, H, W] where D should equal 5 (target window)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # Shape: [B, 128, 4, 12, 12]
        x = self.dropout3d(x)
        
        # Reduce channels from 128 to 64: new shape [B, 64, 4, 12, 12]
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))
        
        b, c, d, h, w = x.shape  # Here, d should be 4 (or your configured depth)
        # Flatten spatial dimensions per slice: shape becomes [B, d, c*h*w] = [B, d, 64*12*12]
        x = x.view(b, d, -1)
        
        # LSTM aggregator: output shape [B, d, 512] (bidirectional LSTM)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)
        
        # Compute attention scores per slice and aggregate to get one feature vector per bag (per study)
        attn_scores = self.attention(lstm_out).squeeze(-1)  # shape: [B, d]
        attn_weights = F.softmax(attn_scores, dim=1)  # shape: [B, d]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # shape: [B, 512]
        
        logits = self.fc(context)
        return logits