import torch


class LTBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, amount):
        x = latent["samples"].clone()
        x = x + amount
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -0.9, "max": 2.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, amount):
        x = latent["samples"].clone()
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * (1.0 + amount) + mean
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTExposure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, factor):
        x = latent["samples"].clone()
        x = x * factor
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTGamma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, gamma):
        x = latent["samples"].clone()
        # Normalize each channel to [0,1] using its min/max, apply gamma, then back
        min_per = x.amin(dim=(2, 3), keepdim=True)
        max_per = x.amax(dim=(2, 3), keepdim=True)
        x_norm = (x - min_per) / (max_per - min_per + 1e-8)
        x_norm = x_norm ** (1.0 / gamma)
        x = x_norm * (max_per - min_per) + min_per
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTInvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent):
        x = latent["samples"].clone()
        x = -x
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "min_val": (
                    "FLOAT",
                    {"default": -3.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "max_val": (
                    "FLOAT",
                    {"default": 3.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, min_val, max_val):
        x = latent["samples"].clone()
        x = torch.clamp(x, min_val, max_val)
        return ({"samples": x, "downscale_ratio_spacial": 8},)


class LTLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "in_black": (
                    "FLOAT",
                    {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "in_white": (
                    "FLOAT",
                    {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "out_black": (
                    "FLOAT",
                    {"default": -2.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "out_white": (
                    "FLOAT",
                    {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, in_black, in_white, out_black, out_white):
        x = latent["samples"].clone()
        x = torch.clamp(x, in_black, in_white)
        x = (x - in_black) / (in_white - in_black + 1e-8)
        x = x * (out_white - out_black) + out_black
        x = torch.clamp(x, -10.0, 10.0)
        return ({"samples": x, "downscale_ratio_spacial": 8},)
