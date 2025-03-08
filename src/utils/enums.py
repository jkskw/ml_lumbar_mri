from enum import Enum

class ClassificationMode(Enum):
    SINGLE_MULTICLASS = "single_multiclass"
    SINGLE_BINARY = "single_binary"
    MULTI_MULTICLASS = "multi_multiclass"
    MULTI_BINARY = "multi_binary"

class ModelArchitecture(Enum):
    SIMPLE_3DCNN = "simple_3dcnn"
    ADVANCED_SINGLE = "advanced_single"
    ADVANCED_SINGLE_IMPROVED = "advanced_single_improved"
    ADVANCED_MULTI = "advanced_multi"
    RESNET3D = "resnet3d"