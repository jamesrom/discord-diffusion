from math import ceil
import torch
import asyncio
import time
import concurrent.futures
from typing import Callable, Awaitable, Any
from dataclasses import dataclass
from diffusion import StableDiffusion
from symbol import while_stmt

progress_bar_length = 20

@dataclass
class QueueState:
    position: int
    progress: float

    def description(self):
        if self.position > 0:
            return f"Queued ({self.position} ahead)"
        else:
            filled = max(ceil(self.progress * progress_bar_length), 1)
            unfilled = progress_bar_length - filled
            return (filled * "▰") + (unfilled * "▱")

background_tasks = set()

class DiffusionWorkerQueue:
    def __init__(self, instance: StableDiffusion):
        self.instance = instance
        self.lock = asyncio.Lock()
        self.queue_size = 0
        self.new_job_started = asyncio.Event()

    async def run(self, *args, progress_callback: Callable[[QueueState], Awaitable[Any]], **kwargs):
        loop = asyncio.get_event_loop()
        position = self.queue_size
        self.queue_size += 1

        # coroutine updates the status when a new job is initiated
        async def progress_coro(pos):
            while pos >= 0:
                await progress_callback(QueueState(position=pos, progress=0))
                await self.new_job_started.wait()
                self.new_job_started.clear()
                pos -= 1

        task = asyncio.create_task(progress_coro(position))
        background_tasks.add(task)

        # instance level lock to ensure only one pipeline job runs at a time
        async with self.lock:
            # notify other waiters that the next job has started (they've progressed in the queue)
            self.new_job_started.set()

            t = None
            def _cb(step: int, timestep: int, latents: torch.FloatTensor):
                # throttle
                nonlocal t
                now = time.time()
                if t is None or now - t > 3:
                    future = asyncio.run_coroutine_threadsafe(progress_callback(QueueState(position=0, progress=step/50)), loop)
                    future.result() # uncomment to block pipeline while discord is notified
                    t = now

            kwargs['callback'] = _cb

            # closure over args, run_in_executor doesn't support kwargs
            def run_sync():
                return self.instance.text2img(*args, **kwargs)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, run_sync)

            # cleanup
            self.queue_size -= 1
            background_tasks.remove(task)

            return result
