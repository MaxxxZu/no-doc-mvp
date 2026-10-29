"""App-owned models with process-wide CPU policy and fail-fast inference admission."""

import logging
from contextlib import contextmanager
from threading import Condition, Lock
from time import perf_counter
from typing import Callable, Iterator

from config import Settings
from services.model_loader import Operation, Pipeline, load_pipeline

logger = logging.getLogger(__name__)
_INFERENCE_SLOT = Lock()
_THREAD_LOCK = Lock()
_CPU_THREADS: int | None = None


class RuntimeUnavailable(Exception):
    pass


class RuntimeBusy(Exception):
    pass


def configure_cpu_threads(threads: int) -> None:
    """Set intra-op threads once per process, before loading/inference.

    Inter-op settings retain PyTorch defaults; they cannot safely be reset after
    parallel work starts. Multiple apps must agree on this process-wide policy.
    """
    global _CPU_THREADS
    with _THREAD_LOCK:
        if _CPU_THREADS is not None:
            if _CPU_THREADS != threads:
                raise RuntimeError("CPU thread policy conflicts with the running process")
            return
        import torch

        torch.set_num_threads(threads)
        _CPU_THREADS = threads


class ModelRuntime:
    def __init__(
        self, settings: Settings,
        loader: Callable[[Operation, Settings], Pipeline] | None = None,
    ) -> None:
        self.settings = settings
        self._loader = loader or load_pipeline
        self._condition = Condition()
        self._active = 0
        self._ready = False
        self._closed = False
        self.classifier: Pipeline | None = None
        self.ner: Pipeline | None = None

    @property
    def ready(self) -> bool:
        # A cheap snapshot, including while startup holds the lifecycle lock.
        return self._ready

    def initialize(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Runtime is closed")
            if self._ready:
                return
            try:
                configure_cpu_threads(self.settings.cpu_threads)
                for operation in ("classify", "ner"):
                    started = perf_counter()
                    logger.info("model_load_started operation=%s", operation)
                    loaded = self._loader(operation, self.settings)
                    if operation == "classify":
                        self.classifier = loaded
                    else:
                        self.ner = loaded
                    loaded = None
                    logger.info("model_load_completed operation=%s duration_seconds=%.6f",
                                operation, perf_counter() - started)
                self._ready = True
                logger.info("runtime_ready")
            except BaseException:
                self.classifier = self.ner = None
                self._ready = False
                logger.error("runtime_initialization_failed")
                # Do not expose arbitrary loader exception text through startup logs.
                raise RuntimeError("Required model initialization failed") from None

    @contextmanager
    def inference(self, operation: Operation) -> Iterator[None]:
        if not self._ready:
            raise RuntimeUnavailable()
        if not _INFERENCE_SLOT.acquire(blocking=False):
            logger.warning("inference_overload operation=%s", operation)
            raise RuntimeBusy()
        admitted = False
        started = perf_counter()
        try:
            with self._condition:
                if not self._ready:
                    raise RuntimeUnavailable()
                self._active += 1
                admitted = True
            try:
                yield
            except BaseException:
                logger.error("inference_failed operation=%s", operation)
                raise
            finally:
                logger.info("inference_finished operation=%s duration_seconds=%.6f",
                            operation, perf_counter() - started)
        finally:
            if admitted:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()
            _INFERENCE_SLOT.release()

    def close(self) -> None:
        with self._condition:
            self._ready = False
            self._closed = True
            # A running CPU call must finish before its references are released.
            while self._active:
                self._condition.wait()
            self.classifier = self.ner = None
            logger.info("runtime_closed")
