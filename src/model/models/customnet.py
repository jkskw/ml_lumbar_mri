import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # pip install timm

class CustomNet(nn.Module):
    """
    Adapted from the RSNA 2024 winning approach: 
      - 5-slice input (1 channel)
      - 2D backbone (ConvNeXt, EfficientNet, etc.)
      - Bi-LSTM + attention-based MIL
      - Single-disease, multi-class (3 classes).
    
    Expects input shape: [B, 1, 5, H, W] from SingleDiseaseDataset 
    where '5' is the # of slices (depth).
    """

    def __init__(self, 
                 num_classes=3, 
                 input_channels=1, 
                 dropout_prob=0.5,
                 backbone="convnext_small", 
                 pretrained=True):
        """
        :param num_classes: 3 for single_multiclass (Normal/Mild, Moderate, Severe)
        :param input_channels: should be 1 (grayscale). 
        :param backbone: timm model name for the 2D encoder
        :param pretrained: if True, load pretrained weights
        :param dropout_prob: dropout used in LSTM or final FC if desired
        """
        super().__init__()

        # 1) 2D encoder from timm
        #   - e.g. 'convnext_small' or 'efficientnetv2_s', etc.
        #   - set in_chans=1 so it accepts grayscale
        self.encoder = timm.create_model(backbone, 
                                         pretrained=pretrained, 
                                         in_chans=input_channels, 
                                         num_classes=0)
        feat_dim = self.encoder.num_features  # output dimension

        # 2) Bi-LSTM to handle the 5 slices
        self.lstm = nn.LSTM(feat_dim, feat_dim // 2, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout_prob)
        # Because it's bidirectional, the LSTM output dimension = feat_dim

        # 3) Attention head
        self.attn_fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(feat_dim, 1) 
        )

        # 4) Final classification
        self.classifier = nn.Linear(feat_dim, num_classes)

        # Optionally, you can add dropout on final:
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x shape => [B, 1, 5, H, W]
        Steps:
        1) Flatten (B*5, 1, H, W)
        2) 2D encoder => (B*5, feats, H_enc, W_enc)
        3) Apply global average pooling => (B*5, feats, 1, 1)
        4) Flatten => (B*5, feats)
        5) Reshape => (B, 5, feats)
        6) Bi-LSTM => (B, 5, feats)
        7) Attention-based MIL => (B, feats)
        8) Classifier => (B, num_classes)
        """
        B, C, D, H, W = x.shape  # e.g., B=16, C=1, D=5, H=128, W=128

        # Flatten depth into batch: [B, D, C, H, W] -> [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # 1) Pass through 2D encoder
        feats = self.encoder.forward_features(x)  # shape: [B*D, feat_dim, H_enc, W_enc]
        
        # 2) Global average pooling to reduce H_enc and W_enc to 1
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))  # shape: [B*D, feat_dim, 1, 1]
        feats = feats.flatten(1)  # shape: [B*D, feat_dim]; feat_dim should now be 768

        # 3) Reshape to [B, D, feat_dim]
        feats = feats.view(B, D, -1)

        # 4) Pass through Bi-LSTM
        lstm_out, _ = self.lstm(feats)  # shape: [B, D, feat_dim]

        # 5) Attention: compute attention scores and weighted average over slices
        attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # shape: [B, D]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # shape: [B, D, 1]
        weighted_sum = torch.sum(lstm_out * attn_weights, dim=1)  # shape: [B, feat_dim]

        # 6) Final classification with dropout
        fused = self.dropout(weighted_sum)
        logits = self.classifier(fused)  # shape: [B, num_classes]

        return logits