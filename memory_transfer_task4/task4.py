#!/usr/bin/env python3
"""CSS 314 - Task 4: CPU IPC pipe vs zero-copy shared-memory overhead.

Runs on macOS, Linux, or Windows using only the Python standard library.
Writes task4_output.txt next to this file with REAL measured timings.
"""

from array import array
from multiprocessing import Pipe, Process, shared_memory, freeze_support
from pathlib import Path
import platform
import statistics
import time

SIZES = (100_000, 1_000_000, 10_000_000)
TRIALS = 5


def pipe_worker(connection):
    """Receive the serialized array itself, then acknowledge it."""
    try:
        while True:
            payload = connection.recv()
            if payload is None:
                break
            connection.send((len(payload), payload[0], payload[-1]))
    finally:
        connection.close()


def shared_worker(connection, name, length):
    """Attach to the same physical buffer; receive only a small go signal."""
    region = shared_memory.SharedMemory(name=name)
    numbers = region.buf[:length * 4].cast("i")
    try:
        while True:
            command = connection.recv()
            if command == "STOP":
                break
            connection.send((len(numbers), numbers[0], numbers[-1]))
    finally:
        numbers.release()
        region.close()
        connection.close()


def run_pipe(source, expected):
    parent, child = Pipe(duplex=True)
    process = Process(target=pipe_worker, args=(child,))
    process.start()
    child.close()
    samples = []
    try:
        for trial in range(TRIALS + 1):  # First iteration warms up IPC.
            start = time.perf_counter_ns()
            parent.send(source)   # Pickle serialization + OS pipe transfer.
            reply = parent.recv()  # Wait until receiver has the payload.
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            if reply != expected:
                raise AssertionError(f"Pipe verification failed: {reply!r}")
            if trial:
                samples.append(elapsed_ms)
        parent.send(None)
    finally:
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Pipe worker exited with status {process.exitcode}")
    return statistics.median(samples)


def run_shared(source, expected):
    # One-time allocation and population are setup, not per-transfer overhead.
    region = shared_memory.SharedMemory(create=True, size=len(source) * source.itemsize)
    parent = child = process = None
    try:
        view = region.buf[:len(source) * source.itemsize].cast("i")
        try:
            view[:] = source
        finally:
            view.release()
        parent, child = Pipe(duplex=True)
        process = Process(
            target=shared_worker, args=(child, region.name, len(source))
        )
        process.start()
        child.close()
        child = None
        samples = []
        for trial in range(TRIALS + 1):
            start = time.perf_counter_ns()
            parent.send("GO")  # Metadata / signal only: no array serialization.
            reply = parent.recv()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            if reply != expected:
                raise AssertionError(f"Shared-memory verification failed: {reply!r}")
            if trial:
                samples.append(elapsed_ms)
        parent.send("STOP")
        return statistics.median(samples)
    finally:
        if child is not None:
            child.close()
        if parent is not None:
            parent.close()
        if process is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        region.close()
        region.unlink()


def main():
    if array("i").itemsize != 4:
        raise RuntimeError("This benchmark requires 32-bit C integers")
    lines = [
        "TASK 4 - Host Memory Transfer / IPC Overhead (CPU)",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        "Array type: signed 32-bit int (4 bytes/element)",
        f"Trials: {TRIALS} measured + 1 warm-up; statistic: median",
        "Metric: request-to-acknowledgement latency (includes small acknowledgement)",
        "Pipe: pickle-serialize full array, transfer, receive and acknowledge",
        "Shared: parent/child already map the same buffer; signal, read and acknowledge",
        "Shared-memory allocation + initial population are excluded (one-time setup)",
        "",
        f"{'N':>12} {'Size (MiB)':>12} {'Pipe (ms)':>13} {'Shared (ms)':>14} {'Ratio':>10}",
        "-" * 67,
    ]
    for n in SIZES:
        source = array("i", range(n))
        expected = (n, 0, n - 1)
        pipe_ms = run_pipe(source, expected)
        shared_ms = run_shared(source, expected)
        ratio = pipe_ms / shared_ms if shared_ms else float("inf")
        lines.append(
            f"{n:12,d} {len(source) * source.itemsize / (1024 * 1024):12.2f} "
            f"{pipe_ms:13.3f} {shared_ms:14.3f} {ratio:9.1f}x"
        )
        del source
    lines += [
        "",
        "Conclusion: A Pipe transfers and deserializes all N integers for each request.",
        "Shared memory shares an existing buffer without copying it per request.",
        "A true end-to-end pipeline must separately account for shared-buffer setup",
        "and any required producer writes; these values isolate transfer/access overhead.",
    ]
    content = "\n".join(lines) + "\n"
    print(content, end="")
    output_path = Path(__file__).with_name("task4_output.txt")
    output_path.write_text(content, encoding="utf-8")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    freeze_support()
    main()
