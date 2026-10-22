import os

# Restrict any numerical libraries to one internal thread per worker.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import math
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103081
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95
TIMED_RUNS = 5


def amdahl_bound(workers: int) -> float:
    """Amdahl's theoretical speedup for a fixed 95% parallel portion."""
    return 1.0 / ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / workers)


def sum_chunk(chunk: np.ndarray) -> float:
    """Sum one disjoint float64 slice on a worker thread."""
    return float(np.sum(chunk, dtype=np.float64))


def benchmark(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Return median measured elapsed ms and sum for a worker count."""
    length = data.size
    chunks = [
        data[(i * length) // workers : ((i + 1) * length) // workers]
        for i in range(workers)
    ]

    def run_sum(pool: ThreadPoolExecutor) -> float:
        # Work is truly concurrent because NumPy's numeric sum releases GIL.
        partial_sums = pool.map(sum_chunk, chunks)
        return math.fsum(partial_sums)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        run_sum(pool)  # Warm up and start all worker threads before timing.
        samples_ms = []
        for _ in range(TIMED_RUNS):
            start_ns = time.perf_counter_ns()
            total = run_sum(pool)
            samples_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

    return statistics.median(samples_ms), total


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = [1, 2, 4, max_cores, max_cores * 2]

    # Same generator, seed and dtype as Task 1; allocation is not timed.
    data = np.random.default_rng(STUDENT_ID).random(ARRAY_SIZE, dtype=np.float64)
    serial_reference = float(np.sum(data, dtype=np.float64))

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / Platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores (os.cpu_count): {max_cores}")
    print(f"Array size: {data.size:,}")
    print(f"Array dtype: {data.dtype}")
    print("Parallel execution: ThreadPoolExecutor + NumPy float64 sum")
    print(f"Timing: median of {TIMED_RUNS} runs; warm-up and pool startup excluded")
    print("Amdahl parallel fraction P: 0.95")
    print()
    print(f"{'Cores':>7} | {'Wall Time (ms)':>14} | {'Speedup':>9} | {'Amdahl Upper Bound':>18}")
    print("-" * 60)

    baseline_ms = None
    for workers in worker_counts:
        elapsed_ms, result = benchmark(data, workers)
        if not math.isclose(result, serial_reference, rel_tol=1e-12, abs_tol=1e-6):
            raise RuntimeError(f"Incorrect sum at {workers} workers: {result}")
        if baseline_ms is None:
            baseline_ms = elapsed_ms
        speedup = baseline_ms / elapsed_ms
        print(f"{workers:7d} | {elapsed_ms:14.3f} | {speedup:9.3f} | {amdahl_bound(workers):18.3f}")

    print()
    print(f"Verified sum: {serial_reference:.10f}")
    print("Result: all worker counts produced the same sum (within float64 tolerance).")


if __name__ == "__main__":
    main()
