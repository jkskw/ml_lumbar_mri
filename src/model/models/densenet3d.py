import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """
    Dense layer with bottleneck structure.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
    Dense block containing multiple dense layers.
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):
    """
    Transition layer between dense blocks.
    """
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    """
    3D DenseNet model for medical image analysis.
    
    Args:
        depth (int): Model depth - 121, 169, 201, or 264
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (1 for single MRI sequence, 2+ for multiple)
        growth_rate (int): Growth rate (k in paper)
        dropout_prob (float): Dropout probability
        bn_size (int): Bottleneck size (factor for number of bottleneck layers)
        sample_size (int): Input spatial dimension
    """
    def __init__(self, depth=121, num_classes=3, input_channels=1, growth_rate=32, 
                 dropout_prob=0.3, bn_size=4, sample_size=112):
        super().__init__()
        
        # Determine block configuration based on depth
        if depth == 121:
            block_config = (6, 12, 24, 16)
        elif depth == 169:
            block_config = (6, 12, 32, 32)
        elif depth == 201:
            block_config = (6, 12, 48, 32)
        elif depth == 264:
            block_config = (6, 12, 64, 48)
        else:
            raise ValueError(f"Unsupported DenseNet3D depth: {depth}. Use 121, 169, 201, or 264")
        
        # Initial convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(input_channels, 64, 
                               kernel_size=(7, 7, 7), 
                               stride=(2, 2, 2), 
                               padding=(3, 3, 3), 
                               bias=False)),
            ('norm0', nn.BatchNorm3d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Dense blocks and transitions
        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_prob
            )
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            # Add transition after each dense block except the last one
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2
                
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(features)
        out = out.view(features.size(0), -1)
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        out = self.classifier(out)
        return out


# Factory functions for different model depths
def DenseNet3D_121(**kwargs):
    """Constructs a DenseNet3D-121 model."""
    return DenseNet3D(depth=121, **kwargs)

def DenseNet3D_169(**kwargs):
    """Constructs a DenseNet3D-169 model."""
    return DenseNet3D(depth=169, **kwargs)

def DenseNet3D_201(**kwargs):
    """Constructs a DenseNet3D-201 model."""
    return DenseNet3D(depth=201, **kwargs)

def DenseNet3D_264(**kwargs):
    """Constructs a DenseNet3D-264 model."""
    return DenseNet3D(depth=264, **kwargs)