import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2,2)
        
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2,2)
        
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.AdaptiveAvgPool3d((1,1,1))
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(128, num_classes)  # num_classes can be 1 if binary, or 3 if multiclass

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class AdvancedSpinal3DNetSingle(nn.Module):
    """
    For single-disease classification (multiclass or binary).
    If binary, pass num_classes=1.
    """
    def __init__(self, num_classes=3, input_channels=1, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((2,2,2))

        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.dropout3d = nn.Dropout3d(dropout_prob)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.AdaptiveAvgPool3d((2,12,12))

        self.lstm = nn.LSTM(128*12*12, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.attention = nn.Sequential(
            nn.Linear(512, 128),  # 256*2 => 512
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc_shared = nn.Linear(512, 256)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, in_channels=1, D, H, W]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3d(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        b, c, d, h, w = x.shape
        x = x.permute(0,2,1,3,4).reshape(b, d, -1)  # [B, D, 128*12*12]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)

        attn_scores = self.attention(lstm_out).squeeze(-1)  # [B, D]
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, 512]

        feat = F.relu(self.fc_shared(context))  # [B, 256]
        out = self.fc_out(feat)                 # [B, num_classes]
        return out


class AdvancedSpinal3DNetImproved(nn.Module):
    """
    Single disease, but improved pooling scheme.
    If binary => num_classes=1
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
        x = self.dropout3d(x)

        b, c, d, h, w = x.shape
        x = x.view(b, d, -1)  # [B, D, 128*12*12]

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, 512]

        out = self.fc(context)  # [B, num_classes]
        return out


class AdvancedSpinal3DNetMulti(nn.Module):
    """
    Multi-disease. If classification_mode='multi_multiclass', each head => 3 outputs
                   if classification_mode='multi_binary',   each head => 1 output
    """
    def __init__(self, input_channels=3, dropout_prob=0.3, output_mode="multi_multiclass"):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d((1,2,2))

        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d((1,2,2))

        self.dropout3d = nn.Dropout3d(dropout_prob)

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(128)
        self.pool3 = nn.AdaptiveAvgPool3d((1,12,12))

        self.lstm = nn.LSTM(128*12*12, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc_shared = nn.Linear(512, 256)

        if output_mode == "multi_multiclass":
            self.fc_scs  = nn.Linear(256, 3)
            self.fc_lnfn = nn.Linear(256, 3)
            self.fc_rnfn = nn.Linear(256, 3)
        else:
            # "multi_binary" => each disease outputs a single logit
            self.fc_scs  = nn.Linear(256, 1)
            self.fc_lnfn = nn.Linear(256, 1)
            self.fc_rnfn = nn.Linear(256, 1)

        self.output_mode = output_mode

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3d(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        b, c, d, h, w = x.shape
        x = x.permute(0,2,1,3,4).reshape(b, d, -1)  # [B, D, features]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [B, 512]

        feat = F.relu(self.fc_shared(context))
        out_scs  = self.fc_scs(feat)
        out_lnfn = self.fc_lnfn(feat)
        out_rnfn = self.fc_rnfn(feat)
        return out_scs, out_lnfn, out_rnfn


class Basic3DResidualBlock(nn.Module):
    """
    A basic 3D residual block:
      - conv3d -> bn3d -> relu
      - conv3d -> bn3d
      + skip
      -> relu
    Optionally downsampling with stride=2 if needed.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout3d(p=dropout_prob) if dropout_prob>0 else nn.Identity()

        # If in/out channels or stride mismatch, we do a conv-based skip
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.dropout(out)  # optional dropout

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out
    
class ResNet3D(nn.Module):
    """
    A simplified 3D ResNet-like architecture.
    For multi-disease, we do 3 heads each with num_classes=3 if 'multi_multiclass'.
    Otherwise just do a single FC if single or 'multi_binary'.

    Example usage:
      model = ResNet3D(input_channels=3, block_channels=[32,64,128], 
                       num_blocks=[2,2,2], 
                       classification_mode="multi_multiclass", 
                       dropout_prob=0.3)

    """
    def __init__(self,
                 input_channels=3,
                 block_channels=(32,64,128),
                 num_blocks=(2,2,2),
                 classification_mode="multi_multiclass",
                 dropout_prob=0.3):
        super().__init__()

        self.classification_mode = classification_mode
        # If multi_multiclass => 3 heads each with 3 classes
        # If multi_binary => 3 heads each with 1 output
        # If single_multiclass => 1 head with e.g. 3 classes
        # If single_binary => 1 head with 1 output

        # Initial stem
        self.conv1 = nn.Conv3d(input_channels, block_channels[0], 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(block_channels[0])
        self.relu  = nn.ReLU(inplace=True)
        # Optional max pool
        self.pool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Make residual layers
        self.layer1 = self._make_layer(block_channels[0], block_channels[0], 
                                       num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block_channels[0], block_channels[1], 
                                       num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block_channels[1], block_channels[2], 
                                       num_blocks[2], stride=2, dropout_prob=dropout_prob)

        # After that, we do a global average pool in 3D
        self.dropout = nn.Dropout3d(dropout_prob)
        
        # For multi disease:
        # If classification_mode = "multi_multiclass", each disease => 3 classes => 3 heads of fc
        # If "multi_binary", each disease => 1 => 3 heads
        # If single => 1 head
        if classification_mode == "multi_multiclass":
            self.fc_scs  = nn.Linear(block_channels[-1], 3)
            self.fc_lnfn = nn.Linear(block_channels[-1], 3)
            self.fc_rnfn = nn.Linear(block_channels[-1], 3)
        elif classification_mode == "multi_binary":
            self.fc_scs  = nn.Linear(block_channels[-1], 1)
            self.fc_lnfn = nn.Linear(block_channels[-1], 1)
            self.fc_rnfn = nn.Linear(block_channels[-1], 1)
        else:
            # single multiclass or single binary
            if classification_mode == "single_multiclass":
                num_out = 3  # or 2, etc.
            else:
                num_out = 1
            self.fc_single = nn.Linear(block_channels[-1], num_out)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1, dropout_prob=0.0):
        """
        Create a stack of `blocks` Basic3DResidualBlock. 
        The first block can have stride=2 if we want downsampling.
        """
        layers = []
        layers.append(Basic3DResidualBlock(in_ch, out_ch, stride=stride, dropout_prob=dropout_prob))
        for _ in range(1, blocks):
            layers.append(Basic3DResidualBlock(out_ch, out_ch, stride=1, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: [B, input_channels, D, H, W]
        x = self.conv1(x)     # => [B, block_channels[0], D/2, H/2, W/2] (approx, from stride=2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)      # => reduce again

        x = self.layer1(x)    # => shape depends on stride=1 in layer1
        x = self.layer2(x)    # => stride=2 => reduce dimension
        x = self.layer3(x)    # => stride=2 => reduce dimension

        # Suppose now x ~ [B, block_channels[-1], Dsmall, Hsmall, Wsmall]
        # global average pool over (D,H,W)
        x = F.adaptive_avg_pool3d(x, (1,1,1))  # => [B, C, 1,1,1]
        x = x.view(x.size(0), -1)              # => [B, C]
        x = self.dropout(x)

        # If multi => 3 heads
        if self.classification_mode == "multi_multiclass":
            out_scs  = self.fc_scs(x)   # => [B,3]
            out_lnfn = self.fc_lnfn(x)  # => [B,3]
            out_rnfn = self.fc_rnfn(x)  # => [B,3]
            return out_scs, out_lnfn, out_rnfn
        elif self.classification_mode == "multi_binary":
            out_scs  = self.fc_scs(x)   # => [B,1]
            out_lnfn = self.fc_lnfn(x)  # => [B,1]
            out_rnfn = self.fc_rnfn(x)  # => [B,1]
            return out_scs, out_lnfn, out_rnfn
        else:
            # single => one FC
            return self.fc_single(x)
