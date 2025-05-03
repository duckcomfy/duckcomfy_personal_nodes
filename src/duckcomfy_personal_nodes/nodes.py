import comfy.sd

# compatibility with efficiency-nodes
EFFICIENCY_ONLY_SCHEDULERS = ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]
VANILLA_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS
EFFICIENCY_SCHEDULERS = VANILLA_SCHEDULERS + EFFICIENCY_ONLY_SCHEDULERS
DETAILER_SCHEDULERS = ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal', 'AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan']

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
}
