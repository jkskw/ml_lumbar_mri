import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvNeXtSmallLSTM(nn.Module):
    """
    A model that:
      1. Uses a pretrained ConvNeXt-Small (via timm) as a 2D feature extractor.
      2. Processes the resulting feature vectors in a Bi-LSTM.
      3. Employs an attention mechanism over the Bi-LSTM outputs.
      4. Outputs 3-class logits (for multi-class tasks).
    
    Expected input shape: [B, 1, D, H, W]
      - B = batch size
      - 1 = single channel
      - D = # of slices (e.g., 5)
      - H, W = spatial dimensions
    """
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super().__init__()
        
        # 1) Create a pretrained ConvNeXt-Small with single-channel input
        self.encoder = timm.create_model(
            "convnext_small",
            pretrained=True,
            in_chans=input_channels,   # single-channel input
            num_classes=0              # we'll ignore the final classifier
        )
        # The "num_features" property in timm often indicates the feature dimension
        self.feature_dim = self.encoder.num_features
        
        # 2) Bi-LSTM
        # We'll do a single-layer, bidirectional LSTM.
        # If you prefer 2 layers, adjust num_layers=2 accordingly.
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, 
            hidden_size=self.feature_dim // 2, 
            num_layers=1,            # or 2
            batch_first=True, 
            bidirectional=True
        )
        
        # 3) Attention head
        # We have a bidirectional LSTM => output dimension = feature_dim
        self.attn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.feature_dim, 1)  # scalar score per time-step
        )
        
        # 4) Classification
        # The final dimension from the attention-pooled vector is self.feature_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Optional dropout to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x shape: [B, 1, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        # ---- (A) Flatten the 'D' dimension into the batch dimension ----
        # So we treat each slice as a separate image in the batch.
        # We'll reorder from [B, C, D, H, W] => [B, D, C, H, W] => [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        
        # ---- (B) Extract 2D features via ConvNeXt-Small ----
        # Forward pass through the model's "forward_features" or the entire encoder with no classifier
        feats_2d = self.encoder.forward_features(x)  
        # feats_2d shape: [B*D, feature_dim, H_enc, W_enc]
        
        # Global average pooling => shape: [B*D, feature_dim]
        feats_2d = F.adaptive_avg_pool2d(feats_2d, (1, 1)).squeeze(-1).squeeze(-1)
        
        # ---- (C) Reshape back to [B, D, feature_dim] for LSTM ----
        feats_2d = feats_2d.view(B, D, self.feature_dim)
        
        # ---- (D) Bi-LSTM ----
        # Output shape => [B, D, feature_dim] (since it's bidirectional with hidden_size=feature_dim//2)
        lstm_out, _ = self.lstm(feats_2d)
        
        # ---- (E) Attention: compute a scalar score for each of the D steps, 
        #      then do a weighted sum (like a soft attention) ----
        attn_scores = self.attn(lstm_out).squeeze(-1)  # [B, D]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, D, 1]
        
        # Weighted sum => [B, feature_dim]
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # ---- (F) Classification head ----
        context = self.dropout(context)
        logits = self.classifier(context)
        
        return logits
