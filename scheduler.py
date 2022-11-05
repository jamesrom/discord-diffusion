import multiprocessing
import threading
import queue
import asyncio
import uuid
from diffusion import StableDiffusionProcess, ShutdownSignal
import dto

MSG_WAIT_TIMEOUT = 120 # Wait up to 2 minutes for a status update
QueueProgressed = 0xC00010FF

class Scheduler:
    def __init__(self, *args, **kwargs):
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        self.args = args
        self.kwargs = kwargs
        self.subscribers = {}
        self.kill = False

    async def schedule(self, *args, **kwargs):
        q = asyncio.Queue()
        pos = self.in_queue.qsize()

        # Create a reference so that we can send messages to the right place
        work_id = uuid.uuid4()
        self.subscribers[work_id] = q

        self.in_queue.put(dto.Job(work_id, *args, **kwargs))
        print(f"A new job has been queued at position {pos}.")
        yield dto.Status(work_id, pos, 0.0, None)

        while True:
            msg = await asyncio.wait_for(q.get(), timeout=MSG_WAIT_TIMEOUT)

            # If queue has progressed, decrement our counter
            if msg == QueueProgressed:
                pos -= 1
                yield dto.Status(work_id, pos, 0.0, None)
                continue

            yield msg
            if msg != QueueProgressed and (msg.images is not None or msg.has_errored):
                break

        del self.subscribers[work_id] # unsubscribe from cat facts

    def __enter__(self):
        self.restart_process()
        self.thread = threading.Thread(target=self._run_background, args=(asyncio.get_event_loop(),))
        self.thread.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.kill = True
        self.in_queue.put(ShutdownSignal)

        self.process.join()
        print('Stable Diffusion process complete.')
        self.thread.join()
        print('Background thread complete.')

    def restart_process(self):
        self.process = StableDiffusionProcess(self.in_queue, self.out_queue, *self.args, **self.kwargs)
        self.process.start()

    def _run_background(self, loop):
        print("Scheduler thread is running.")
        while not self.kill:
            if not self.process.is_alive():
                self.restart_process()

            try:
                status = self.out_queue.get(block=True, timeout=3)
            except queue.Empty:
                continue

            q = self.subscribers.get(status.work_id)
            if q is not None:
                loop.call_soon_threadsafe(q.put_nowait, status)

            # If work complete, notify all subscribers that their position has changed
            if status.images is not None or status.has_errored:
                for id, sub in self.subscribers.items():
                    loop.call_soon_threadsafe(sub.put_nowait, QueueProgressed)
