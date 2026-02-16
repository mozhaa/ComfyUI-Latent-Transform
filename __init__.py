from .basic import (
    LTAdd,
    LTBlend,
    LTBlur,
    LTGaussianNoise,
    LTMultiply,
    LTSharpen,
    LTWave,
)

NODE_CLASS_MAPPINGS = {
    "LT: Multiply": LTMultiply,
    "LT: Add": LTAdd,
    "LT: Blur": LTBlur,
    "LT: Sharpen": LTSharpen,
    "LT: Gaussian Noise": LTGaussianNoise,
    "LT: Blend": LTBlend,
    "LT: Wave": LTWave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LT: Multiply": "LT: Multiply",
    "LT: Add": "LT: Add",
    "LT: Blur": "LT: Blur",
    "LT: Sharpen": "LT: Sharpen",
    "LT: Gaussian Noise": "LT: Gaussian Noise",
    "LT: Blend": "LT: Blend",
    "LT: Wave": "LT: Wave",
}
