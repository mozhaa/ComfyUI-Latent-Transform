import math

import torch


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


class LTChannelTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "m00": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m01": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m02": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m10": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m11": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m12": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m20": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m21": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
                "m22": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        B, C, H, W = x.shape
        if C >= 3:
            matrix = torch.tensor(
                [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], device=x.device
            )
            rgb = x[:, :3, :, :]
            rgb_flat = rgb.view(B, 3, -1)
            rgb_new = torch.einsum("ij,bjn->bin", matrix, rgb_flat)
            x[:, :3, :, :] = rgb_new.view(B, 3, H, W)
        new_latent["samples"] = x
        return (new_latent,)


class LTHueShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "angle_deg": (
                    "FLOAT",
                    {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, angle_deg):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        B, C, H, W = x.shape
        if C >= 3:
            angle = math.radians(angle_deg)
            u = math.cos(angle)
            w = math.sin(angle)
            a = (1.0 - u) / 3.0
            m00 = u + a
            m01 = a - w / math.sqrt(3.0)
            m02 = a + w / math.sqrt(3.0)
            m10 = a + w / math.sqrt(3.0)
            m11 = u + a
            m12 = a - w / math.sqrt(3.0)
            m20 = a - w / math.sqrt(3.0)
            m21 = a + w / math.sqrt(3.0)
            m22 = u + a
            matrix = torch.tensor(
                [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], device=x.device
            )
            rgb = x[:, :3, :, :]
            rgb_flat = rgb.view(B, 3, -1)
            rgb_new = torch.einsum("ij,bjn->bin", matrix, rgb_flat)
            x[:, :3, :, :] = rgb_new.view(B, 3, H, W)
        new_latent["samples"] = x
        return (new_latent,)
