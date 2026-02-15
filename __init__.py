from .basic import (
    LatentTransformBrightness,
    LatentTransformClamp,
    LatentTransformContrast,
    LatentTransformExposure,
    LatentTransformGamma,
    LatentTransformInvert,
    LatentTransformLevels,
)

NODE_CLASS_MAPPINGS = {
    "Latent Transform: Brightness": LatentTransformBrightness,
    "Latent Transform: Contrast": LatentTransformContrast,
    "Latent Transform: Exposure": LatentTransformExposure,
    "Latent Transform: Gamma": LatentTransformGamma,
    "Latent Transform: Invert": LatentTransformInvert,
    "Latent Transform: Clamp": LatentTransformClamp,
    "Latent Transform: Levels": LatentTransformLevels,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Latent Transform: Brightness": "Latent Transform: Brightness",
    "Latent Transform: Contrast": "Latent Transform: Contrast",
    "Latent Transform: Exposure": "Latent Transform: Exposure",
    "Latent Transform: Gamma": "Latent Transform: Gamma",
    "Latent Transform: Invert": "Latent Transform: Invert",
    "Latent Transform: Clamp": "Latent Transform: Clamp",
    "Latent Transform: Levels": "Latent Transform: Levels",
}
