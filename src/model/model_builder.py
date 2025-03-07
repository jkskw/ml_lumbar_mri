import torch.nn as nn
from src.model.models import (
    Simple3DCNN,
    AdvancedSpinal3DNetSingle,
    AdvancedSpinal3DNetImproved,
    AdvancedSpinal3DNetMulti,
    ResNet3D
)

def build_model(config, in_channels):
    classification_mode = config["training"]["classification_mode"]
    model_arch = config["training"]["model_arch"]
    dropout_prob = config["training"]["dropout_prob"]

    if classification_mode.startswith("single"):
        num_classes = 3 if classification_mode == "single_multiclass" else 1
        if model_arch == "simple_3dcnn":
            model = Simple3DCNN(num_classes=num_classes, input_channels=in_channels, dropout_prob=dropout_prob)
        elif model_arch == "advanced_single":
            model = AdvancedSpinal3DNetSingle(num_classes=num_classes, input_channels=in_channels, dropout_prob=dropout_prob)
        elif model_arch == "advanced_single_improved":
            model = AdvancedSpinal3DNetImproved(num_classes=num_classes, input_channels=in_channels, dropout_prob=dropout_prob)
        else:
            model = Simple3DCNN(num_classes=num_classes, input_channels=in_channels, dropout_prob=dropout_prob)
        criterion = nn.CrossEntropyLoss() if classification_mode == "single_multiclass" else nn.BCEWithLogitsLoss()
    else:
        if "multiclass" in classification_mode:
            if model_arch == "resnet3d":
                model = ResNet3D(
                    input_channels=in_channels,
                    block_channels=[32, 64, 128],
                    num_blocks=[2, 2, 2],
                    classification_mode="multi_multiclass",
                    dropout_prob=dropout_prob
                )
            elif model_arch == "advanced_multi":
                model = AdvancedSpinal3DNetMulti(
                    input_channels=in_channels,
                    dropout_prob=dropout_prob,
                    output_mode="multi_multiclass"
                )
            else:
                model = AdvancedSpinal3DNetMulti(
                    input_channels=in_channels,
                    dropout_prob=dropout_prob,
                    output_mode="multi_multiclass"
                )
            criterion = nn.CrossEntropyLoss()
        else:
            model = AdvancedSpinal3DNetMulti(
                input_channels=in_channels,
                dropout_prob=dropout_prob,
                output_mode="multi_binary"
            )
            criterion = nn.BCEWithLogitsLoss()
    return model, criterion
