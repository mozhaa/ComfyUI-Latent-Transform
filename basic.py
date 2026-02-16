import torch


class LTBrightness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -100.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, amount):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        x = x + amount
        new_latent["samples"] = x
        return (new_latent,)


class LTContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -100.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, amount):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * (1.0 + amount) + mean
        new_latent["samples"] = x
        return (new_latent,)


class LTExposure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, factor):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        x = x * factor
        new_latent["samples"] = x
        return (new_latent,)


class LTGamma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, gamma):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        min_per = x.amin(dim=(2, 3), keepdim=True)
        max_per = x.amax(dim=(2, 3), keepdim=True)
        x_norm = (x - min_per) / (max_per - min_per + 1e-8)
        x_norm = x_norm ** (1.0 / gamma)
        x = x_norm * (max_per - min_per) + min_per
        new_latent["samples"] = x
        return (new_latent,)


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
        new_latent = latent.copy()
        x = latent["samples"].clone()
        x = -x
        new_latent["samples"] = x
        return (new_latent,)


class LTClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "min_val": (
                    "FLOAT",
                    {"default": -3.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "max_val": (
                    "FLOAT",
                    {"default": 3.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, min_val, max_val):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        x = torch.clamp(x, min_val, max_val)
        new_latent["samples"] = x
        return (new_latent,)


class LTLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "in_black": (
                    "FLOAT",
                    {"default": -2.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "in_white": (
                    "FLOAT",
                    {"default": 2.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "out_black": (
                    "FLOAT",
                    {"default": -2.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "out_white": (
                    "FLOAT",
                    {"default": 2.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, in_black, in_white, out_black, out_white):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        x = torch.clamp(x, in_black, in_white)
        x = (x - in_black) / (in_white - in_black + 1e-8)
        x = x * (out_white - out_black) + out_black
        new_latent["samples"] = x
        return (new_latent,)
