from dataclasses import dataclass
from typing import List, Optional
import PIL.Image
import uuid

import config

PROGRESS_BAR_LENGTH = 20

@dataclass
class Status:
    work_id: uuid.uuid4
    position: int
    progress: float
    images: List[PIL.Image.Image]
    has_errored: bool=False

    def __str__(self):
        if self.has_errored:
            return f"An error has occured, this message will be deleted shortly."
        if self.position != None and self.position > 0:
            return f"Queued ({self.position} ahead)"
        else:
            filled = max(round(self.progress * PROGRESS_BAR_LENGTH), 1)
            unfilled = PROGRESS_BAR_LENGTH - filled
            return (filled * "▰") + (unfilled * "▱")

@dataclass
class Job:
    work_id: uuid.uuid4        # Internal, the unique ID of the job
    prompt: str                # Description of the image to be generated
    height: int=512            # Image height, must be divisible by 8
    width: int=512             # Image width, must be divisible by 8
    seed: Optional[int]=None   # RNG seed
    sfw: bool=True             # Use safety check
    samples: int=1             # Number of images to create per run
    steps: int=config.DEFAULT_STEPS   # Number of sampling steps
    scale: float=config.DEFAULT_SCALE # Classifier free guidance scale

    def __str__(self):
        return f"{self.work_id}, prompt: {self.prompt}"
