import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

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

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)

class Bottleneck3D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dropout_prob=0.0, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(planes * self.expansion)

        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ResNet3D(nn.Module):
    """
    A simplified 3D ResNet-like architecture.
    This unified ResNet3D accepts:
      - input_channels: number of input channels.
      - block_channels: tuple of channels for each block group.
      - num_blocks: tuple of the number of blocks in each group.
      - num_classes: output dimension (1 for binary, 3 for multiclass).
      - dropout_prob: dropout probability.
      - block_type: either ResidualBlock3D (default) or Bottleneck3D.
    """
    def __init__(self,
                 input_channels=3,
                 block_channels=(32, 64, 128),
                 num_blocks=(2, 2, 2),
                 num_classes=1,
                 dropout_prob=0.3,
                 block_type=ResidualBlock3D):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, block_channels[0],
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_type, block_channels[0], block_channels[0],
                                        num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block_type, block_channels[0], block_channels[1],
                                        num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block_type, block_channels[1], block_channels[2],
                                        num_blocks[2], stride=2, dropout_prob=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)
        # Global average pooling will reduce the spatial dimensions to 1.
        self.fc = nn.Linear(block_channels[-1], num_classes)

    def _make_layer(self, block_type, in_channels, out_channels, blocks, stride=1, dropout_prob=0.0):
        layers = []
        layers.append(block_type(in_channels, out_channels, stride=stride, dropout_prob=dropout_prob))
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, stride=1, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, input_channels, D, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Factory functions:

def ResNet3D_10(num_classes, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        input_channels=input_channels,
        block_channels=(32, 64, 128),
        num_blocks=(1, 1, 1),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        block_type=ResidualBlock3D
    )

def ResNet3D_18(num_classes, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        input_channels=input_channels,
        block_channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        block_type=ResidualBlock3D
    )

def ResNet3D_34(num_classes, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        input_channels=input_channels,
        block_channels=(32, 64, 128),
        num_blocks=(3, 4, 6),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        block_type=ResidualBlock3D
    )

def ResNet3D_50(num_classes, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        input_channels=input_channels,
        block_channels=(64, 128, 256),
        num_blocks=(3, 4, 6),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        block_type=Bottleneck3D
    )

def ResNet3D_101(num_classes, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        input_channels=input_channels,
        block_channels=(64, 128, 256),
        num_blocks=(3, 4, 23),
        num_classes=num_classes,
        dropout_prob=dropout_prob,
        block_type=Bottleneck3D
    )
