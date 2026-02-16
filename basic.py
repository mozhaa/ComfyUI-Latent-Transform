import torch
import torch.nn.functional as F


class LTMultiply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel": (["all", "c0", "c1", "c2", "c3"], {"default": "all"}),
                "factor": (
                    "FLOAT",
                    {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, channel, factor):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        if channel == "all":
            x = x * factor
        else:
            idx = int(channel[1])
            x[:, idx : idx + 1, :, :] = x[:, idx : idx + 1, :, :] * factor
        new_latent["samples"] = x
        return (new_latent,)


class LTAdd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel": (["all", "c0", "c1", "c2", "c3"], {"default": "all"}),
                "amount": (
                    "FLOAT",
                    {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, channel, amount):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        if channel == "all":
            x = x + amount
        else:
            idx = int(channel[1])
            x[:, idx : idx + 1, :, :] = x[:, idx : idx + 1, :, :] + amount
        new_latent["samples"] = x
        return (new_latent,)


class LTBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel": (["all", "c0", "c1", "c2", "c3"], {"default": "all"}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, channel, kernel_size):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        B, C, H, W = x.shape
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (
            kernel_size * kernel_size
        )
        if channel == "all":
            channels = list(range(C))
        else:
            channels = [int(channel[1])]
        for ch in channels:
            inp = x[:, ch : ch + 1, :, :]

            padded = F.pad(
                inp,
                (
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                ),
                mode="reflect",
            )
            blurred = F.conv2d(padded, kernel, padding=0)
            x[:, ch : ch + 1, :, :] = blurred
        new_latent["samples"] = x
        return (new_latent,)


class LTSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel": (["all", "c0", "c1", "c2", "c3"], {"default": "all"}),
                "strength": (
                    "FLOAT",
                    {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.1},
                ),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, channel, strength, kernel_size):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        B, C, H, W = x.shape
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (
            kernel_size * kernel_size
        )
        if channel == "all":
            channels = list(range(C))
        else:
            channels = [int(channel[1])]
        for ch in channels:
            inp = x[:, ch : ch + 1, :, :]
            padded = F.pad(
                inp,
                (
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                    kernel_size // 2,
                ),
                mode="reflect",
            )
            blurred = F.conv2d(padded, kernel, padding=0)
            sharpened = inp + strength * (inp - blurred)
            x[:, ch : ch + 1, :, :] = sharpened
        new_latent["samples"] = x
        return (new_latent,)


class LTGaussianNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "channel": (["all", "c0", "c1", "c2", "c3"], {"default": "all"}),
                "strength": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent, channel, strength, seed):
        new_latent = latent.copy()
        x = latent["samples"].clone()
        torch.manual_seed(seed)
        noise = torch.randn_like(x) * strength
        if channel == "all":
            x = x + noise
        else:
            idx = int(channel[1])
            x[:, idx : idx + 1, :, :] = (
                x[:, idx : idx + 1, :, :] + noise[:, idx : idx + 1, :, :]
            )
        new_latent["samples"] = x
        return (new_latent,)


class LTBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "mode": (
                    ["normal", "multiply", "add", "subtract"],
                    {"default": "normal"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "latent/transform"

    def apply(self, latent_a, latent_b, mode, strength, mask=None):
        new_latent = latent_a.copy()
        a = latent_a["samples"].clone()
        b = latent_b["samples"].clone()

        if a.shape != b.shape:
            b = F.interpolate(b, size=(a.shape[2], a.shape[3]), mode="bilinear")

        B, C, H, W = a.shape

        if mask is not None:
            mask = mask.to(a.device).float()

            if mask.shape[1:] != (H, W):
                mask = F.interpolate(
                    mask.unsqueeze(1), size=(H, W), mode="bilinear"
                ).squeeze(1)

            if mask.shape[0] < B:
                mask = mask.expand(B, -1, -1)
            blend_factor = mask * strength
            blend_factor = torch.clamp(blend_factor, 0.0, 1.0)
            blend_factor = blend_factor.unsqueeze(1)
        else:
            blend_factor = torch.full((B, 1, H, W), strength, device=a.device)

        if mode == "normal":
            blended = b
        elif mode == "multiply":
            blended = a * b
        elif mode == "add":
            blended = a + b
        elif mode == "subtract":
            blended = a - b
        else:
            raise ValueError(f"Invalid blend mode: {mode}")

        result = a * (1 - blend_factor) + blended * blend_factor
        new_latent["samples"] = result
        return (new_latent,)
