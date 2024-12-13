from .dfn_vit import Qwen2VLDFNVisionTransformer
from .siglip import SiglipVisionModel


VISION_TRANSFORMER_CLASSES = {
    'qwen2_vl': Qwen2VLDFNVisionTransformer,
    'siglip': SiglipVisionModel
}
