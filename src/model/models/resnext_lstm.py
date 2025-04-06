import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Provided ResNeXt bottleneck block
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, dropout_prob=0.0):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.dropout is not None:
            out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# Our new architecture using ResNeXt as encoder, followed by Bi-LSTM & attention
class ResNeXtEncoderBiLSTMClassifier(nn.Module):
    def __init__(self, 
                 depth=50, 
                 num_classes=3, 
                 input_channels=1, 
                 dropout_prob=0.3, 
                 cardinality=32,
                 pool_time=4, 
                 pool_spatial=(12, 12)):
        """
        Args:
            depth (int): Depth of ResNeXt (e.g. 50, 101, etc.)
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels.
            dropout_prob (float): Dropout probability.
            cardinality (int): Number of groups in ResNeXt.
            pool_time (int): Output depth after adaptive pooling.
            pool_spatial (tuple): Output (height, width) after pooling.
        """
        super(ResNeXtEncoderBiLSTMClassifier, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        # Initial convolution and pooling (similar to ResNeXt3D)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Define ResNeXt layers based on the provided depth
        if depth == 50:
            layers_cfg = [3, 4, 6, 3]
        elif depth == 101:
            layers_cfg = [3, 4, 23, 3]
        elif depth == 152:
            layers_cfg = [3, 8, 36, 3]
        elif depth == 200:
            layers_cfg = [3, 24, 36, 3]
        else:
            raise ValueError(f"Unsupported ResNeXt depth: {depth}. Use 50, 101, 152, or 200.")

        self.layer1 = self._make_layer(ResNeXtBottleneck, 128, layers_cfg[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 256, layers_cfg[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 512, layers_cfg[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(ResNeXtBottleneck, 1024, layers_cfg[3], stride=2, dropout_prob=dropout_prob)
        # After layer4, output channels = 1024 * expansion (here, 1024*2 = 2048)

        # Reduce channel dimension via a 1x1x1 convolution (from 2048 -> 128)
        self.conv_reduce = nn.Conv3d(1024 * ResNeXtBottleneck.expansion, 128, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm3d(128)
        
        # Adaptive average pooling to get a fixed output: (pool_time, pool_spatial[0], pool_spatial[1])
        self.adaptive_pool = nn.AdaptiveAvgPool3d((pool_time, pool_spatial[0], pool_spatial[1]))
        self.dropout3d = nn.Dropout3d(dropout_prob)

        # Bi-LSTM aggregator
        # The LSTM input dimension equals: 128 * pool_spatial[0] * pool_spatial[1]
        lstm_input_dim = 128 * pool_spatial[0] * pool_spatial[1]
        self.lstm = nn.LSTM(lstm_input_dim, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_prob)

        # Attention mechanism over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Final classification layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample, dropout_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, input_channels, D, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Expected shape: [B, 2048, D_enc, H_enc, W_enc]

        # Reduce channel dimension
        x = self.conv_reduce(x)
        x = self.bn_reduce(x)
        x = self.relu(x)

        # Adaptive pooling to a fixed shape, e.g. [B, 128, pool_time, 12, 12]
        x = self.adaptive_pool(x)
        x = self.dropout3d(x)
        B, C, T, H, W = x.shape

        # Flatten spatial dimensions for each time-step (slice)
        x = x.view(B, T, -1)  # Shape: [B, T, C*H*W]

        # Bi-directional LSTM over the sequence of slices
        lstm_out, _ = self.lstm(x)  # Output shape: [B, T, 512]
        lstm_out = self.dropout_lstm(lstm_out)

        # Attention mechanism: compute scalar scores per time-step
        attn_scores = self.attention(lstm_out).squeeze(-1)  # [B, T]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, T, 1]

        # Aggregate LSTM outputs via weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 512]

        # Final classification
        out = self.fc(context)
        return out

# Example usage:
if __name__ == "__main__":
    # Create a dummy input: batch size 2, single channel, 16 slices, 112x112 spatially.
    dummy_input = torch.randn(2, 1, 16, 112, 112)
    model = ResNeXtEncoderBiLSTMClassifier(depth=50, num_classes=3, input_channels=1, dropout_prob=0.3,
                                             cardinality=32, pool_time=4, pool_spatial=(12, 12))
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [2, 3]
