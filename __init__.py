from .channels import (
    LTAdd,
    LTChannelAdd,
    LTChannelMultiply,
    LTMultiply,
)

NODE_CLASS_MAPPINGS = {
    "LT: Multiply": LTMultiply,
    "LT: Add": LTAdd,
    "LT: Channel Multiply": LTChannelMultiply,
    "LT: Channel Add": LTChannelAdd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LT: Multiply": "LT: Multiply",
    "LT: Add": "LT: Add",
    "LT: Channel Multiply": "LT: Channel Multiply",
    "LT: Channel Add": "LT: Channel Add",
}
