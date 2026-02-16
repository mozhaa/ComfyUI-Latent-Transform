from .basic import (
    LTBrightness,
    LTClamp,
    LTContrast,
    LTExposure,
    LTGamma,
    LTInvert,
    LTLevels,
)
from .channels import (
    LTChannelAdd,
    LTChannelMultiply,
    LTChannelTransform,
    LTHueShift,
)

NODE_CLASS_MAPPINGS = {
    "LT: Brightness": LTBrightness,
    "LT: Contrast": LTContrast,
    "LT: Exposure": LTExposure,
    "LT: Gamma": LTGamma,
    "LT: Invert": LTInvert,
    "LT: Clamp": LTClamp,
    "LT: Levels": LTLevels,
    "LT: Channel Multiply": LTChannelMultiply,
    "LT: Channel Add": LTChannelAdd,
    "LT: Channel Transform": LTChannelTransform,
    "LT: Hue Shift": LTHueShift,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LT: Brightness": "LT: Brightness",
    "LT: Contrast": "LT: Contrast",
    "LT: Exposure": "LT: Exposure",
    "LT: Gamma": "LT: Gamma",
    "LT: Invert": "LT: Invert",
    "LT: Clamp": "LT: Clamp",
    "LT: Levels": "LT: Levels",
    "LT: Channel Multiply": "LT: Channel Multiply",
    "LT: Channel Add": "LT: Channel Add",
    "LT: Channel Transform": "LT: Channel Transform",
    "LT: Hue Shift": "LT: Hue Shift",
}
