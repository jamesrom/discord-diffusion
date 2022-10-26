import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from future.utils import raise_from
import os
from threading import Lock

_device="cuda"

class StableDiffusion:
    def __init__(
        self,
        token: str, # huggingface token
        model: str="CompVis/stable-diffusion-v1-4",
        revision: str="main",
        attention_slicing: bool=False, # false = faster, true = less memory
        default_seed: int=None,
    ):
        self.model = model
        self.revision = revision
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=rev_to_dtype(revision), revision=revision, use_auth_token=token
        ).to(_device)

        if attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

        if default_seed == None:
            self.default_seed = torch.random.seed()
        else:
            self.default_seed = default_seed

        # Save the default safety checker so we can override/replace it query time
        self.default_safety_check = self.pipeline.safety_checker
        self.lock = Lock()

    def text2img(
        self,
        prompt: str,
        height: int=512,
        width: int=512,
        seed: int=None,
        sfw: bool=True,
        samples: int=1, # Number of images to create per run
        iterations: int=1, # Number of times to run pipeline
        steps: int=50, # Number of sampling steps
        scale: float=7.5, # Classifier free guidance scale
        callback=None,
    ):
        with self.lock:
            images = []
            if seed == None:
                seed = self.default_seed
            if sfw:
                self.pipeline.safety_checker = self.default_safety_check
            else:
                self.pipeline.safety_checker = skip_safety_check

            try:
                generator = torch.Generator(device=_device).manual_seed(seed)
                for i in range(iterations):
                    with autocast(_device):
                        result = self.pipeline(
                            [prompt] * samples,
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            guidance_scale=scale,
                            generator=generator,
                            callback=callback
                        )

                    for j, image in enumerate(result.images):
                        # check if image is all-black
                        if not image.getbbox():
                            continue
                        name = "output/%s__steps_%d__scale_%0.2f__seed_%d__n_%d" % (prompt.replace(" ", "_")[:170], steps, scale, seed, j * samples + j + 1)
                        images.append({'name': name, 'image': image})
            except RuntimeError as e:
                if "illegal memory access was encountered" in str(e):
                    raise_from(RuntimeError("GPU out of memory? See README for help."), e)
                else:
                    raise

            return images

def rev_to_dtype(revision: str):
    if revision == "main":
        return torch.float32
    if revision == "fp16":
        return torch.float16
    raise AssertionError("revision must be 'main' or 'fp16'")

def skip_safety_check(images, *args, **kwargs):
    return images, False
