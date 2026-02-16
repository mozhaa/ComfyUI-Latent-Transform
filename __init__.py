from .basic import (
    LTBrightness,
    LTContrast,
    LTExposure,
    LTGamma,
    LTInvert,
    LTClamp,
    LTLevels,
)

NODE_CLASS_MAPPINGS = {
    "LT: Brightness": LTBrightness,
    "LT: Contrast": LTContrast,
    "LT: Exposure": LTExposure,
    "LT: Gamma": LTGamma,
    "LT: Invert": LTInvert,
    "LT: Clamp": LTClamp,
    "LT: Levels": LTLevels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LT: Brightness": "LT: Brightness",
    "LT: Contrast": "LT: Contrast",
    "LT: Exposure": "LT: Exposure",
    "LT: Gamma": "LT: Gamma",
    "LT: Invert": "LT: Invert",
    "LT: Clamp": "LT: Clamp",
    "LT: Levels": "LT: Levels",
}