import comfy.sd

# compatibility with efficiency-nodes
EFFICIENCY_SCHEDULERS = ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS"]
SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + EFFICIENCY_SCHEDULERS
DETAILER_SCHEDULERS = ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform', 'beta', 'linear_quadratic', 'kl_optimal', 'AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', 'LTXV[default]', 'OSS FLUX', 'OSS Wan']

class CastEfficiencySchedulerToDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (SCHEDULERS,),
            }
        }
    RETURN_TYPES = (DETAILER_SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "doit"
    CATEGORY = "duckcomfy"
    def doit(self, scheduler):
        # Convert efficiency scheduler to regular scheduler by defaulting to the first regular scheduler (in comfy.samplers.KSampler.SCHEDULERS) if the scheduler is in EFFICIENCY_SCHEDULERS
        effective_scheduler = scheduler if scheduler in DETAILER_SCHEDULERS else DETAILER_SCHEDULERS[0]
        return (effective_scheduler,)

class ToKSamplerSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (SCHEDULERS,),
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
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "INT", comfy.samplers.KSampler.SAMPLERS, SCHEDULERS, "INT", "FLOAT")
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

NODE_CLASS_MAPPINGS = {
    "CastEfficiencySchedulerToDetailer": CastEfficiencySchedulerToDetailer,
    "ToDuckComfyGlobals": ToDuckComfyGlobals,
    "FromDuckComfyGlobals": FromDuckComfyGlobals,
    "ToKSamplerSettings": ToKSamplerSettings,
    "DuckSeedSelector": DuckSeedSelector,
    "DenoiseSelector": DenoiseSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CastEfficiencySchedulerToDetailer": "Cast Efficiency Scheduler To Detailer",
    "ToDuckComfyGlobals": "To DuckComfy Globals",
    "FromDuckComfyGlobals": "From DuckComfy Globals",
    "ToKSamplerSettings": "To KSampler Settings",
    "DuckSeedSelector": "Duck Seed Selector",
    "DenoiseSelector": "Denoise Selector",
}
