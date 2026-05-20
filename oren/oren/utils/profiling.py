import threading
import time
from typing import ClassVar

import torch
from tqdm import tqdm


class TimerRecords:
    """Global accumulator of timer measurements, keyed by message label.

    Mirrors `erl::common::BlockTimerRecords`: every `CpuTimer` / `GpuTimer` instance with
    `record=True` appends to this registry on exit (after any configured warmup).
    """

    _records: ClassVar[dict[str, dict[str, float]]] = {}
    _labels: ClassVar[list[str]] = []
    _max_label_length: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def add_record(cls, label: str, duration_s: float) -> None:
        """Append a single measurement for `label`. `duration_s` is in seconds."""
        with cls._lock:
            if len(label) > cls._max_label_length:
                cls._max_label_length = len(label)
            record = cls._records.get(label)
            if record is None:
                record = {"current": 0.0, "total": 0.0, "count": 0}
                cls._records[label] = record
                cls._labels.append(label)
                cls._labels.sort()
            record["current"] = duration_s
            record["total"] += duration_s
            record["count"] += 1

    @classmethod
    def print_records(cls) -> None:
        """Print a formatted summary of all recorded timers. Durations shown in milliseconds."""
        with cls._lock:
            if not cls._labels:
                tqdm.write("Timer Records: (empty)")
                return
            label_w = cls._max_label_length + 2
            lines = ["Timer Records:"]
            lines.append(
                f"{'Label':<{label_w}}"
                f"{'Current (ms)':>15}"
                f"{'Mean (ms)':>15}"
                f"{'Total (ms)':>15}"
                f"{'Count':>10}"
            )
            lines.append("-" * (label_w + 15 + 15 + 15 + 10))
            for label in cls._labels:
                rec = cls._records[label]
                count = rec["count"]
                mean = rec["total"] / count if count > 0 else 0.0
                lines.append(
                    f"{label:<{label_w}}"
                    f"{rec['current'] * 1000.0:>15.3f}"
                    f"{mean * 1000.0:>15.3f}"
                    f"{rec['total'] * 1000.0:>15.3f}"
                    f"{count:>10d}"
                )
            tqdm.write("\n".join(lines))

    @classmethod
    def clear(cls) -> None:
        """Drop all accumulated records."""
        with cls._lock:
            cls._records.clear()
            cls._labels.clear()
            cls._max_label_length = 0


def print_records() -> None:
    """Module-level shortcut for `TimerRecords.print_records()`."""
    TimerRecords.print_records()


class CpuTimer:

    def __init__(
        self,
        message,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CPU wall-clock timer context manager.

        Args:
            message: Label printed alongside the measured time.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False, the context manager is a no-op.
            verbose: When True, print the timing summary on every successful exit.
            record: When True, append each post-warmup measurement to `TimerRecords` keyed by `message`.
        """
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.record:
            TimerRecords.add_record(self.message, self.t)
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def cpu_timer(message, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `CpuTimer` context manager.

    Args:
        message: Label passed to the underlying `CpuTimer`.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, timing is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `TimerRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with CpuTimer(message, warmup=warmup, enable=enable, verbose=verbose, record=record):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class GpuTimer:

    def __init__(
        self,
        message,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CUDA-event-based GPU timer context manager.

        Args:
            message: Label printed alongside the measured time.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False, the context manager is a no-op.
            verbose: When True, print the timing summary on every successful exit.
            record: When True, append each post-warmup measurement to `TimerRecords` keyed by `message`.
        """
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.t = self.start.elapsed_time(self.end) / 1e3
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.record:
            TimerRecords.add_record(self.message, self.t)
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def gpu_timer(message, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `GpuTimer` context manager.

    Args:
        message: Label passed to the underlying `GpuTimer`.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, timing is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `TimerRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with GpuTimer(message, warmup=warmup, enable=enable, verbose=verbose, record=record):
                return func(*args, **kwargs)

        return wrapper

    return decorator
