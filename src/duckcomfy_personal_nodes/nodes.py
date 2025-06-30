import comfy.sd
from .attention_couple import (
    AttentionCouple,
    AttentionCoupleRegion,
    AttentionCoupleRegions,
)
import time
import comfy.model_management
import re
import socket
import nodes
import folder_paths
from .functions_upscale import *
import torch


# compatibility with efficiency-nodes
EFFICIENCY_ONLY_SCHEDULERS = ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]
VANILLA_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
EFFICIENCY_SCHEDULERS = VANILLA_SCHEDULERS + EFFICIENCY_ONLY_SCHEDULERS
DETAILER_SCHEDULERS = ['simple', 'sgm_uniform', 'karras', 'exponential', 'ddim_uniform', 'beta', 'normal', 'linear_quadratic', 'kl_optimal', 'AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan']

class CastEfficiencySchedulerToDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (EFFICIENCY_SCHEDULERS,),
            }
        }
    RETURN_TYPES = (DETAILER_SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, scheduler):
        effective_scheduler = scheduler if scheduler in DETAILER_SCHEDULERS else DETAILER_SCHEDULERS[0]
        return (effective_scheduler,)

class CastSchedulerToDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (VANILLA_SCHEDULERS,),
            }
        }
    RETURN_TYPES = (DETAILER_SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, scheduler):
        effective_scheduler = scheduler if scheduler in DETAILER_SCHEDULERS else DETAILER_SCHEDULERS[0]
        return (effective_scheduler,)

class ToKSamplerSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (VANILLA_SCHEDULERS,),
                "steps": ("INT", {"default": 60, "min": 1, "max": 250, "step": 1}),
                "cfg": ("FLOAT", {"default": 6.5, "min": 0, "max": 20, "step": 0.1}),
            }}
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("ksampler_settings",)
    FUNCTION = "doit"
    CATEGORY="duckcomfy"
    def doit(self,seed,sampler_name,scheduler,steps,cfg):
        ksampler_settings = seed, sampler_name, scheduler, steps, cfg
        return(ksampler_settings,)

class ToDuckComfyGlobals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "ksampler_settings": ("BASIC_PIPE",)
            },
        }
    RETURN_TYPES = ("BASIC_PIPE", )
    RETURN_NAMES = ("globals", )
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, model, clip, vae, positive, negative, ksampler_settings):
        pipe = (model, clip, vae, positive, negative, ksampler_settings)
        return (pipe, )

class FromDuckComfyGlobals:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"globals": ("BASIC_PIPE",), }, }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "INT", comfy.samplers.KSampler.SAMPLERS, VANILLA_SCHEDULERS, "INT", "FLOAT")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative", "seed", "sampler_name", "scheduler", "steps", "cfg")
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, globals):
        model, clip, vae, positive, negative, ksampler_settings = globals
        seed, sampler_name, scheduler, steps, cfg = ksampler_settings
        return model, clip, vae, positive, negative, seed, sampler_name, scheduler, steps, cfg

class DuckSeedSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_global_seed": ("BOOLEAN", {"default": True}),
                "global_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, use_global_seed, global_seed, seed):
        effective_seed = global_seed if use_global_seed else seed
        return (effective_seed,)

class DenoiseSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("denoise",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, denoise):
        return (denoise,)

class BatchSizeSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("batch_size",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, batch_size):
        return (batch_size,)

class I2iToggle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_i2i": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("enable_i2i",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, enable_i2i):
        return (enable_i2i,)

class I2Cnet2IToggle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_i2cnet2i": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("enable_i2cnet2i",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, enable_i2cnet2i):
        return (enable_i2cnet2i,)

class CSwitchBooleanConditioning:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": ("CONDITIONING", {"lazy": True}),
                "on_false": ("CONDITIONING", {"lazy": True}),
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "duckcomfy"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    FUNCTION = "execute"

    def check_lazy_status(self, on_true=None, on_false=None, boolean=True):
        needed = "on_true" if boolean else "on_false"
        return [needed]

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

class CSwitchBooleanLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": ("LATENT", {"lazy": True}),
                "on_false": ("LATENT", {"lazy": True}),
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "duckcomfy"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    FUNCTION = "execute"

    def check_lazy_status(self, on_true=None, on_false=None, boolean=True):
        needed = "on_true" if boolean else "on_false"
        return [needed]

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

class CSwitchBooleanFloat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": ("FLOAT", {"lazy": True}),
                "on_false": ("FLOAT", {"lazy": True}),
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "duckcomfy"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)

    FUNCTION = "execute"

    def check_lazy_status(self, on_true=None, on_false=None, boolean=True):
        needed = "on_true" if boolean else "on_false"
        return [needed]

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

class CSwitchBooleanConditioning:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": ("CONDITIONING", {"lazy": True}),
                "on_false": ("CONDITIONING", {"lazy": True}),
                "boolean": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "duckcomfy"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    FUNCTION = "execute"

    def check_lazy_status(self, on_true=None, on_false=None, boolean=True):
        needed = "on_true" if boolean else "on_false"
        return [needed]

    def execute(self, on_true, on_false, boolean=True):
        if boolean:
            return (on_true,)
        else:
            return (on_false,)

class ConditioningFallback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "primary": ("CONDITIONING",),
                "fallback": ("CONDITIONING",)
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "duckcomfy"

    def execute(self, primary=None, fallback=None):
        return (fallback if primary is None else primary,)

class ImageFallback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "primary": ("IMAGE",),
                "fallback": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "duckcomfy"

    def execute(self, primary=None, fallback=None):
        return (fallback if primary is None else primary,)

class ModelFallback:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "primary": ("MODEL",),
                "fallback": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "duckcomfy"

    def execute(self, primary=None, fallback=None):
        return (fallback if primary is None else primary,)

class TwoTextConcat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", "}),
                "clean_whitespace": (["true", "false"],),
            },
            "optional": {
                "text_a": ("STRING", {"forceInput": True}),
                "text_b": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_concatenate"

    CATEGORY = "duckcomfy"

    def text_concatenate(self, delimiter, clean_whitespace, **kwargs):
        text_inputs = []

        # Handle special case where delimiter is "\n" (literal newline).
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str):
                if clean_whitespace == "true":
                    # Remove leading and trailing whitespace around this input.
                    v = v.strip()

                # Only use this input if it's a non-empty string, since it
                # never makes sense to concatenate totally empty inputs.
                # NOTE: If whitespace cleanup is disabled, inputs containing
                # 100% whitespace will be treated as if it's a non-empty input.
                if v != "":
                    text_inputs.append(v)

        # Merge the inputs. Will always generate an output, even if empty.
        merged_text = delimiter.join(text_inputs)

        return (merged_text,)

class IsStringEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STRING",)
            },
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_empty",)
    FUNCTION = "execute"
    CATEGORY = "duckcomfy"
    def execute(self, string):
        return (string == "" or string.isspace(),)

class PromptOverrideSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_override": ("STRING", {"default":"", "placeholder": "Prompt override", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_override",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, val):
        return (val,)

class StringLiteral:
    def __init__(self, ):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "to_string"

    CATEGORY = "duckcomfy"

    def to_string(self, text):
        return (text,)

class imageSize:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("INT", "INT")
  RETURN_NAMES = ("width_int", "height_int")
  OUTPUT_NODE = True
  FUNCTION = "image_width_height"

  CATEGORY = "duckcomfy"

  def image_width_height(self, image):
    _, raw_H, raw_W, _ = image.shape

    width = raw_W
    height = raw_H

    if width is not None and height is not None:
      result = (width, height)
    else:
      result = (0, 0)
    return {"ui": {"text": "Width: "+str(width)+" , Height: "+str(height)}, "result": result}

class SDXL_Resolutions:
    resolution = ["square - 1024x1024 (1:1)","landscape - 1152x896 (4:3)","landscape - 1216x832 (3:2)","landscape - 1344x768 (16:9)","landscape - 1536x640 (21:9)", "portrait - 896x1152 (3:4)","portrait - 832x1216 (2:3)","portrait - 768x1344 (9:16)","portrait - 640x1536 (9:21)"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (s.resolution,),
            }
        }
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolutions"

    CATEGORY="duckcomfy"

    def get_resolutions(self,resolution):
        width = 1024
        height = 1024
        width = int(width)
        height = int(height)
        if(resolution == "square - 1024x1024 (1:1)"):
            width = 1024
            height = 1024
        if(resolution == "landscape - 1152x896 (4:3)"):
            width = 1152
            height = 896
        if(resolution == "landscape - 1216x832 (3:2)"):
            width = 1216
            height = 832
        if(resolution == "landscape - 1344x768 (16:9)"):
            width = 1344
            height = 768
        if(resolution == "landscape - 1536x640 (21:9)"):
            width = 1536
            height = 640
        if(resolution == "portrait - 896x1152 (3:4)"):
            width = 896
            height = 1152
        if(resolution == "portrait - 832x1216 (2:3)"):
            width = 832
            height = 1216
        if(resolution == "portrait - 768x1344 (9:16)"):
            width = 768
            height = 1344
        if(resolution == "portrait - 640x1536 (9:21)"):
            width = 640
            height = 1536

        return(int(width),int(height))

class Wan480_Resolutions:
    resolution = ["square - 512x512","landscape - 832x480 (16:9)","portrait - 480x832 (9:16)"]
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (s.resolution,),
            }
        }
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolutions"
    CATEGORY="duckcomfy"
    def get_resolutions(self,resolution):
        width = 512
        height = 512
        width = int(width)
        height = int(height)
        if(resolution == "square - 512x512"):
            width = 512
            height = 512
        if(resolution == "landscape - 832x480 (16:9)"):
            width = 832
            height = 480
        if(resolution == "portrait - 480x832 (9:16)"):
            width = 480
            height = 832
        return(int(width),int(height))

class WAS_Text_Concatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": ", "}),
                "clean_whitespace": (["true", "false"],),
            },
            "optional": {
                "text_a": ("STRING", {"forceInput": True}),
                "text_b": ("STRING", {"forceInput": True}),
                "text_c": ("STRING", {"forceInput": True}),
                "text_d": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_concatenate"

    CATEGORY = "duckcomfy"

    def text_concatenate(self, delimiter, clean_whitespace, **kwargs):
        text_inputs = []

        # Handle special case where delimiter is "\n" (literal newline).
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str):
                if clean_whitespace == "true":
                    # Remove leading and trailing whitespace around this input.
                    v = v.strip()

                # Only use this input if it's a non-empty string, since it
                # never makes sense to concatenate totally empty inputs.
                # NOTE: If whitespace cleanup is disabled, inputs containing
                # 100% whitespace will be treated as if it's a non-empty input.
                if v != "":
                    text_inputs.append(v)

        # Merge the inputs. Will always generate an output, even if empty.
        merged_text = delimiter.join(text_inputs)

        return (merged_text,)

class WAS_Text_to_Conditioning:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "text_to_conditioning"

    CATEGORY = "duckcomfy"

    def text_to_conditioning(self, clip, text):
        encoder = nodes.CLIPTextEncode()
        encoded = encoder.encode(clip=clip, text=text)
        return (encoded[0], { "ui": { "string": text } })

class CR_UpscaleImage:
    @classmethod
    def INPUT_TYPES(s):

        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]

        return {"required":
                    {"image": ("IMAGE",),
                     "upscale_model": (folder_paths.get_filename_list("upscale_models"), ),
                     "mode": (["rescale", "resize"],),
                     "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01}),
                     "resize_width": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                     "resampling_method": (resampling_methods,),
                     "supersample": (["true", "false"],),
                     "rounding_modulus": ("INT", {"default": 8, "min": 8, "max": 1024, "step": 8}),
                     }
                }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "upscale"
    CATEGORY = "duckcomfy"

    def upscale(self, image, upscale_model, rounding_modulus=8, loops=1, mode="rescale", supersample='true', resampling_method="lanczos", rescale_factor=2, resize_width=1024):

        # Load upscale model
        up_model = load_model(upscale_model)

        # Upscale with model
        up_image = upscale_with_model(up_model, image)

        for img in image:
            pil_img = tensor2pil(img)
            original_width, original_height = pil_img.size

        for img in up_image:
            # Get new size
            pil_img = tensor2pil(img)
            upscaled_width, upscaled_height = pil_img.size

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Upscale-Nodes#cr-upscale-image"

        # Return if no rescale needed
        if upscaled_width == original_width and rescale_factor == 1:
            return (up_image, show_help)

        # Image resize
        scaled_images = []

        for img in up_image:
            scaled_images.append(pil2tensor(apply_resize_image(tensor2pil(img), original_width, original_height, rounding_modulus, mode, supersample, rescale_factor, resize_width, resampling_method)))
        images_out = torch.cat(scaled_images, dim=0)

        return (images_out, show_help, )


class isMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "duckcomfy"

    def execute(self, mask):
        if mask is None:
            return (True,)
        if torch.all(mask == 0):
            return (True,)
        return (False,)

class ImageReroute:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("IMAGE", )
  RETURN_NAMES = ("IMAGE", )
  FUNCTION = "doit"

  CATEGORY = "duckcomfy"

  def doit(self, image):
      return (image,)


NODE_CLASS_MAPPINGS = {
    "CastEfficiencySchedulerToDetailer": CastEfficiencySchedulerToDetailer,
    "CastSchedulerToDetailer": CastSchedulerToDetailer,
    "ToDuckComfyGlobals": ToDuckComfyGlobals,
    "FromDuckComfyGlobals": FromDuckComfyGlobals,
    "ToKSamplerSettings": ToKSamplerSettings,
    "DuckSeedSelector": DuckSeedSelector,
    "DenoiseSelector": DenoiseSelector,
    "BatchSizeSelector": BatchSizeSelector,
    "I2iToggle": I2iToggle,
    "I2Cnet2IToggle": I2Cnet2IToggle,
    "CSwitchBooleanConditioning": CSwitchBooleanConditioning,
    "CSwitchBooleanLatent": CSwitchBooleanLatent,
    "CSwitchBooleanFloat": CSwitchBooleanFloat,
    "ConditioningFallback": ConditioningFallback,
    "ModelFallback": ModelFallback,
    "TwoTextConcat": TwoTextConcat,
    "IsStringEmpty": IsStringEmpty,
    "PromptOverrideSelector": PromptOverrideSelector,
    "DuckAttentionCouple": AttentionCouple,
    "DuckAttentionCoupleRegion": AttentionCoupleRegion,
    "DuckAttentionCoupleRegions": AttentionCoupleRegions,
    "DuckTextMultiline": StringLiteral,
    "imageSize": imageSize,
    "SDXL_Resolutions": SDXL_Resolutions,
    "Duck_Text_Concatenate": WAS_Text_Concatenate,
    "Duck_Text_to_Conditioning": WAS_Text_to_Conditioning,
    "CR_UpscaleImage": CR_UpscaleImage,
    "isMaskEmpty": isMaskEmpty,
    "ImageReroute": ImageReroute,
    "Wan480_Resolutions": Wan480_Resolutions,
    "ImageFallback": ImageFallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CastEfficiencySchedulerToDetailer": "Cast Efficiency Scheduler To Detailer",
    "CastSchedulerToDetailer": "Cast Scheduler To Detailer",
    "ToDuckComfyGlobals": "To DuckComfy Globals",
    "FromDuckComfyGlobals": "From DuckComfy Globals",
    "ToKSamplerSettings": "To KSampler Settings",
    "DuckSeedSelector": "Duck Seed Selector",
    "DenoiseSelector": "Denoise Selector",
    "BatchSizeSelector": "Batch Size Selector",
    "I2iToggle": "I2i Toggle",
    "I2Cnet2IToggle": "I2Cnet2i Toggle",
    "CSwitchBooleanConditioning": "C Switch Boolean Conditioning",
    "CSwitchBooleanLatent": "C Switch Boolean Latent",
    "CSwitchBooleanFloat": "C Switch Boolean Float",
    "ConditioningFallback": "Conditioning Fallback",
    "ModelFallback": "Model Fallback",
    "TwoTextConcat": "Two Text Concat",
    "IsStringEmpty": "Is String Empty",
    "PromptOverrideSelector": "Prompt Override Selector",
    "DuckAttentionCouple": "DuckAttention Couple",
    "DuckAttentionCoupleRegion": "DuckAttention Couple Region",
    "DuckAttentionCoupleRegions": "DuckAttention Couple Regions",
    "imageSize": "Image Size",
    "SDXL_Resolutions": "SDXL Resolutions",
    "CR_UpscaleImage": "CR Upscale Image",
    "isMaskEmpty": "Is Mask Empty",
    "Duck_Text_Concatenate": "Duck Text Concatenate",
    "Duck_Text_to_Conditioning": "Duck Text to Conditioning",
    "DuckTextMultiline": "Duck Text Multiline",
    "ImageReroute": "Image Reroute",
    "Wan480_Resolutions": "Wan480 Resolutions",
    "ImageFallback": "Image Fallback",
}
