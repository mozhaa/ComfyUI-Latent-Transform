from .basic import (
    LTAdd,
    LTBlend,
    LTBlur,
    LTGaussianNoise,
    LTMultiply,
    LTSharpen,
)

NODE_CLASS_MAPPINGS = {
    "LT: Multiply": LTMultiply,
    "LT: Add": LTAdd,
    "LT: Blur": LTBlur,
    "LT: Sharpen": LTSharpen,
    "LT: Gaussian Noise": LTGaussianNoise,
    "LT: Blend": LTBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LT: Multiply": "LT: Multiply",
    "LT: Add": "LT: Add",
    "LT: Blur": "LT: Blur",
    "LT: Sharpen": "LT: Sharpen",
    "LT: Gaussian Noise": "LT: Gaussian Noise",
    "LT: Blend": "LT: Blend",
}
