import torch


class LTMultiply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
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


class LTAdd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "amount": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
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


class LTChannelMultiply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "c0": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c1": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c2": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c3": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, c0, c1, c2, c3):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        factors = torch.tensor([c0, c1, c2, c3], device=x.device).view(1, -1, 1, 1)
        x = x * factors
        new_latent["samples"] = x
        return (new_latent,)


class LTChannelAdd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "c0": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c1": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c2": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "c3": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, c0, c1, c2, c3):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        biases = torch.tensor([c0, c1, c2, c3], device=x.device).view(1, -1, 1, 1)
        x = x + biases
        new_latent["samples"] = x
        return (new_latent,)
