import torch.nn as nn
import importlib

from src.utils.enums import ClassificationMode
from src.utils.registry import MODEL_REGISTRY

def load_model(model_name, **kwargs):
    """
    Dynamically loads a model class from the registered module.
    
    :param model_name: Name of the model class (as defined in the registry).
    :param kwargs: Additional arguments to pass to the model constructor.
    :return: Instantiated model object.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered. Available models: {list(MODEL_REGISTRY.keys())}")

    module_path = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name)

    return model_class(**kwargs)


def build_model(config: dict, in_channels: int):
    """
    Constructs a model based on a given configuration.
    
    :param config: Dictionary containing model parameters.
    :param in_channels: Number of input channels (e.g., 1 for T1/T2 separately, 2 for combined).
    :return: Instantiated model and loss criterion.
    """
    model_name = config["training"]["model_arch"]
    classification_mode = ClassificationMode(config["training"]["classification_mode"])
    dropout_prob = config["training"]["dropout_prob"]

    num_classes = 3 if classification_mode == ClassificationMode.SINGLE_MULTICLASS else 1

    # Load model dynamically
    model = load_model(
        model_name=model_name,
        num_classes=num_classes,
        input_channels=in_channels,
        dropout_prob=dropout_prob
    )

    # Set loss function
    if classification_mode == ClassificationMode.SINGLE_MULTICLASS:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    return model, criterion
