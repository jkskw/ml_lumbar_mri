import torch.nn as nn

from src.model.models import (
    Simple3DCNN,
    AdvancedSpinal3DNetSingle,
    AdvancedSpinal3DNetImproved,
    AdvancedSpinal3DNetMulti,
    ResNet3D
)
from src.utils.enums import ClassificationMode, ModelArchitecture


def _get_model_and_criterion(
    classification_mode: ClassificationMode,
    model_arch: ModelArchitecture,
    input_channels: int,
    dropout_prob: float
):
    """
    Internal helper to build the correct model + criterion given classification mode and architecture.
    """
    if classification_mode in [ClassificationMode.SINGLE_MULTICLASS, ClassificationMode.SINGLE_BINARY]:
        # Single-disease
        num_classes = 3 if classification_mode == ClassificationMode.SINGLE_MULTICLASS else 1

        # Pick model architecture
        if model_arch == ModelArchitecture.SIMPLE_3DCNN:
            model = Simple3DCNN(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout_prob=dropout_prob
            )
        elif model_arch == ModelArchitecture.ADVANCED_SINGLE:
            model = AdvancedSpinal3DNetSingle(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout_prob=dropout_prob
            )
        elif model_arch == ModelArchitecture.ADVANCED_SINGLE_IMPROVED:
            model = AdvancedSpinal3DNetImproved(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout_prob=dropout_prob
            )
        else:
            # Fallback
            model = Simple3DCNN(
                num_classes=num_classes,
                input_channels=input_channels,
                dropout_prob=dropout_prob
            )

        # Criterion
        if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
            criterion = nn.CrossEntropyLoss()
        else:  # SINGLE_BINARY
            criterion = nn.BCEWithLogitsLoss()

    else:
        # Multi-disease
        # The model will produce 3 heads, each either multiclass(3) or binary(1)
        if model_arch == ModelArchitecture.RESNET3D:
            model = ResNet3D(
                input_channels=input_channels,
                block_channels=[32, 64, 128],
                num_blocks=[2, 2, 2],
                classification_mode=classification_mode.value,
                dropout_prob=dropout_prob
            )
        elif model_arch == ModelArchitecture.ADVANCED_MULTI:
            output_mode = "multi_multiclass" if classification_mode == ClassificationMode.MULTI_MULTICLASS else "multi_binary"
            model = AdvancedSpinal3DNetMulti(
                input_channels=input_channels,
                dropout_prob=dropout_prob,
                output_mode=output_mode
            )
        else:
            # Fallback to advanced multi
            output_mode = "multi_multiclass" if classification_mode == ClassificationMode.MULTI_MULTICLASS else "multi_binary"
            model = AdvancedSpinal3DNetMulti(
                input_channels=input_channels,
                dropout_prob=dropout_prob,
                output_mode=output_mode
            )

        # Criterion
        if classification_mode == ClassificationMode.MULTI_MULTICLASS:
            criterion = nn.CrossEntropyLoss()
        else:  # MULTI_BINARY
            criterion = nn.BCEWithLogitsLoss()

    return model, criterion


def build_model(config: dict, in_channels: int):
    """
    Reads classification_mode, model_arch, dropout_prob from config
    and returns (model, criterion).
    """

    classification_str = config["training"]["classification_mode"]
    model_arch_str = config["training"]["model_arch"]
    dropout_prob = config["training"]["dropout_prob"]

    # Convert string to Enum
    classification_mode = ClassificationMode(classification_str)
    model_arch = ModelArchitecture(model_arch_str)

    model, criterion = _get_model_and_criterion(
        classification_mode=classification_mode,
        model_arch=model_arch,
        input_channels=in_channels,
        dropout_prob=dropout_prob
    )
    return model, criterion