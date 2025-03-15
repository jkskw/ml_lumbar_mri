from enum import Enum
from src.utils.registry import MODEL_REGISTRY

class ClassificationMode(Enum):
    SINGLE_MULTICLASS = "single_multiclass"
    SINGLE_BINARY = "single_binary"
    MULTI_MULTICLASS = "multi_multiclass"
    MULTI_BINARY = "multi_binary"

ModelArchitecture = Enum("ModelArchitecture", {name: name for name in MODEL_REGISTRY.keys()})
