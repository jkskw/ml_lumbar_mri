import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

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

class CustomResNet3D(nn.Module):
    """
    Flexible 3D‐ResNet with BasicBlock3D or Bottleneck3D, for single/multi.
    Replaced nn.Dropout3d with nn.Dropout so no 2D shape warnings appear.
    """
    def __init__(self,
                 block,
                 layers,
                 classification_mode,
                 input_channels=1,
                 dropout_prob=0.3,
                 base_filters=64):
        super().__init__()
        self.classification_mode = classification_mode
        self.inplanes = base_filters

        # Stem
        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(base_filters)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Layers
        self.layer1 = self._make_layer(block, base_filters,   layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_filters*8, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        # *** Use normal dropout after flatten
        self.dropout = nn.Dropout(p=dropout_prob)

        final_channels = base_filters * 8 * block.expansion
        if classification_mode == "multi_multiclass":
            self.fc_scs  = nn.Linear(final_channels, 3)
            self.fc_lnfn = nn.Linear(final_channels, 3)
            self.fc_rnfn = nn.Linear(final_channels, 3)
        elif classification_mode == "multi_binary":
            self.fc_scs  = nn.Linear(final_channels, 1)
            self.fc_lnfn = nn.Linear(final_channels, 1)
            self.fc_rnfn = nn.Linear(final_channels, 1)
        else:
            if classification_mode == "single_multiclass":
                num_out = 3
            else:
                num_out = 1
            self.fc_single = nn.Linear(final_channels, num_out)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, block, planes, blocks, stride=1):
        out_channels = planes * block.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers_list = []
        layers_list.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # shape => [B, final_channels]
        x = self.dropout(x)

        if self.classification_mode == "multi_multiclass":
            return self.fc_scs(x), self.fc_lnfn(x), self.fc_rnfn(x)
        elif self.classification_mode == "multi_binary":
            return self.fc_scs(x), self.fc_lnfn(x), self.fc_rnfn(x)
        else:
            return self.fc_single(x)
