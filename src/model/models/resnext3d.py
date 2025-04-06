import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

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
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
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


class ResNeXt3D(nn.Module):
    """
    3D ResNeXt model for medical image analysis.
    
    Args:
        depth (int): Model depth - 50, 101, 152, or 200
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for single MRI sequence, 2+ for multiple)
        dropout_prob (float): Dropout probability
        cardinality (int): ResNeXt cardinality (number of grouped convolutions)
        sample_size (int): Input spatial dimension
        sample_duration (int): Input temporal dimension
    """
    def __init__(self, depth=50, num_classes=3, input_channels=1, dropout_prob=0.3, 
                 cardinality=32, sample_size=112, sample_duration=16):
        self.inplanes = 64
        super(ResNeXt3D, self).__init__()
        
        # Parameters specific to ResNeXt
        self.cardinality = cardinality
        
        # First convolution layer
        self.conv1 = nn.Conv3d(
            input_channels,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        
        # ResNeXt layers
        if depth == 50:
            self.layers = [3, 4, 6, 3]
        elif depth == 101:
            self.layers = [3, 4, 23, 3]
        elif depth == 152:
            self.layers = [3, 8, 36, 3]
        elif depth == 200:
            self.layers = [3, 24, 36, 3]
        else:
            raise ValueError(f"Unsupported ResNeXt3D depth: {depth}. Use 50, 101, 152, or 200.")
        
        self.layer1 = self._make_layer(ResNeXtBottleneck, 128, self.layers[0], 
                                      stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 256, self.layers[1], 
                                      stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 512, self.layers[2], 
                                      stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(ResNeXtBottleneck, 1024, self.layers[3], 
                                      stride=2, dropout_prob=dropout_prob)
        
        # Calculate adaptive pooling output size
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification layer
        self.fc = nn.Linear(1024 * ResNeXtBottleneck.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample, dropout_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)

        return x


# Factory functions for different model depths
def ResNeXt3D_50(**kwargs):
    """Constructs a ResNeXt3D-50 model."""
    return ResNeXt3D(depth=50, **kwargs)

def ResNeXt3D_101(**kwargs):
    """Constructs a ResNeXt3D-101 model."""
    return ResNeXt3D(depth=101, **kwargs)

def ResNeXt3D_152(**kwargs):
    """Constructs a ResNeXt3D-152 model."""
    return ResNeXt3D(depth=152, **kwargs)

def ResNeXt3D_200(**kwargs):
    """Constructs a ResNeXt3D-200 model."""
    return ResNeXt3D(depth=200, **kwargs)