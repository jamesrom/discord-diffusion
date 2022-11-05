from multiprocessing import Process, Queue

from diffusers import StableDiffusionPipeline
from future.utils import raise_from
from torch import autocast
import torch

import dto

_device="cuda"
ShutdownSignal = 0xDEADFEED

class StableDiffusionProcess(Process):
    def __init__(
        self,
        in_queue: Queue,
        out_queue: Queue,
        token: str, # huggingface token
        model: str="CompVis/stable-diffusion-v1-4",
        revision: str="main",
        attention_slicing: bool=False,
        default_seed: int=None,
    ):
        super(StableDiffusionProcess, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.token = token
        self.model = model
        self.revision = revision
        self.attention_slicing = attention_slicing

        if default_seed == None:
            self.default_seed = torch.random.seed()
        else:
            self.default_seed = default_seed

        print("Isolated Stable Diffusion process started.")

    # Initializes the pipeline which will automatically download required model from huggingface
    def setup(self):
        print('Initializing pipeline, this may take a few seconds to a few minutes depending on cache availability...')
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model, torch_dtype=rev_to_dtype(self.revision), revision=self.revision, use_auth_token=self.token
        ).to(_device)

        if self.attention_slicing:
            self.pipeline.enable_attention_slicing()
        else:
            self.pipeline.disable_attention_slicing()

        # Save the default safety checker so we can override/replace it query time
        self.default_safety_check = self.pipeline.safety_checker
        print("Pipeline initialized. Ready for work.")


    def run(self):
        self.setup()
        try:
            self.handled_run()
        except KeyboardInterrupt:
            print('Waiting for process to exit. If you wish to force shutdown now, press CTRL+D.')
            self.in_queue.put(ShutdownSignal)

    def handled_run(self):
        while True:
            job = self.in_queue.get() # blocks until there's work to be done or shutdown
            if job == ShutdownSignal:
                return # The only way out is through - Virgil

            seed = job.seed if job.seed is not None else self.default_seed

            if job.sfw:
                self.pipeline.safety_checker = self.default_safety_check
            else:
                self.pipeline.safety_checker = skip_safety_check

            def _cb(step: int, timestep: int, latents: torch.FloatTensor):
                self.out_queue.put_nowait(dto.Status(job.work_id, None, step/job.steps, None, False))

            try:
                generator = torch.Generator(device=_device).manual_seed(seed)
                with autocast(_device):
                    result = self.pipeline(
                        [job.prompt] * job.samples,
                        height=job.height,
                        width=job.width,
                        num_inference_steps=job.steps,
                        guidance_scale=job.scale,
                        generator=generator,
                        callback=_cb
                    )

                    # image.getbbox() returns truthy if the image is not all black
                    images = [image for image in result.images if image.getbbox()]

                self.out_queue.put(dto.Status(job.work_id, None, 1.0, images=images))

            except RuntimeError as e:
                if "illegal memory access was encountered" in str(e):
                    raise_from(RuntimeError("GPU out of memory? See README for help."), e)
                else:
                    self.out_queue.put(dto.Status(job.work_id, None, None, None, True))
                    raise

            except:
                self.out_queue.put(dto.Status(job.work_id, None, None, None, True))
                raise


def rev_to_dtype(revision: str):
    if revision == "main":
        return torch.float32
    if revision == "fp16":
        return torch.float16
    raise AssertionError("revision must be 'main' or 'fp16'")

def skip_safety_check(images, *args, **kwargs):
    return images, False
