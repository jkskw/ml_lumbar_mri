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

        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob>0 else nn.Identity()

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
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(planes * 4)

        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

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
    For multi-disease => 3 heads, single => 1 head.
    We use nn.Dropout after flattening to avoid warnings.
    """
    def __init__(self,
                 input_channels=3,
                 block_channels=(32,64,128),
                 num_blocks=(2,2,2),
                 classification_mode="multi_multiclass",
                 dropout_prob=0.3):
        super().__init__()
        self.classification_mode = classification_mode

        # Stem
        self.conv1 = nn.Conv3d(input_channels, block_channels[0], 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(block_channels[0])
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block_channels[0], block_channels[0],
                                       num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block_channels[0], block_channels[1],
                                       num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block_channels[1], block_channels[2],
                                       num_blocks[2], stride=2, dropout_prob=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

        # If multi => 3 heads
        if classification_mode == "multi_multiclass":
            self.fc_scs  = nn.Linear(block_channels[-1], 3)
            self.fc_lnfn = nn.Linear(block_channels[-1], 3)
            self.fc_rnfn = nn.Linear(block_channels[-1], 3)
        elif classification_mode == "multi_binary":
            self.fc_scs  = nn.Linear(block_channels[-1], 1)
            self.fc_lnfn = nn.Linear(block_channels[-1], 1)
            self.fc_rnfn = nn.Linear(block_channels[-1], 1)
        else:
            # single
            if classification_mode == "single_multiclass":
                num_out = 3
            else:
                num_out = 1
            self.fc_single = nn.Linear(block_channels[-1], num_out)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1, dropout_prob=0.0):
        layers = []
        layers.append(ResidualBlock3D(in_ch, out_ch,
                                           stride=stride,
                                           dropout_prob=dropout_prob))
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_ch, out_ch,
                                               stride=1,
                                               dropout_prob=dropout_prob))
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

        # global avg pool => shape [B, block_channels[-1], 1,1,1]
        x = F.adaptive_avg_pool3d(x, (1,1,1))
        x = x.view(x.size(0), -1)  # => [B, block_channels[-1]]

        x = self.dropout(x)  # normal dropout

        if self.classification_mode.startswith("multi_"):
            out_scs  = self.fc_scs(x)
            out_lnfn = self.fc_lnfn(x)
            out_rnfn = self.fc_rnfn(x)
            return out_scs, out_lnfn, out_rnfn
        else:
            return self.fc_single(x)

def ResNet3D_10(classification_mode, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        block=ResidualBlock3D,
        layers=[1,1,1,1],
        classification_mode=classification_mode,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        base_filters=64
    )

def ResNet3D_18(classification_mode, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        block=ResidualBlock3D,
        layers=[2,2,2,2],
        classification_mode=classification_mode,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        base_filters=64
    )

def ResNet3D_34(classification_mode, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        block=ResidualBlock3D,
        layers=[3,4,6,3],
        classification_mode=classification_mode,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        base_filters=64
    )

def ResNet3D_50(classification_mode, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        block=Bottleneck3D,
        layers=[3,4,6,3],
        classification_mode=classification_mode,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        base_filters=64
    )

def ResNet3D_101(classification_mode, input_channels=1, dropout_prob=0.3):
    return ResNet3D(
        block=Bottleneck3D,
        layers=[3,4,23,3],
        classification_mode=classification_mode,
        input_channels=input_channels,
        dropout_prob=dropout_prob,
        base_filters=64
    )