"""Task 1: hardware probe and single-threaded serial baseline.

Requires: numpy (python3 -m pip install numpy)
Run: python3 task1.py | tee task1_output.txt
"""

import os

# Set the thread limits before importing NumPy.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import platform
import subprocess
import time

import numpy as np


STUDENT_ID = 230103081
ARRAY_SIZE = 20_000_000


def cpu_details() -> str:
    """Return a usable CPU or platform description on macOS/Linux/Windows."""
    if platform.system() == "Darwin":
        for field in ("machdep.cpu.brand_string", "hw.model"):
            try:
                name = subprocess.check_output(
                    ["sysctl", "-n", field],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if name:
                    return name
            except (OSError, subprocess.CalledProcessError):
                pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as cpu_file:
                for line in cpu_file:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def main() -> None:
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / Platform: {cpu_details()} ({platform.machine()})")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")

    # Generate the data deterministically; generation is excluded from timing.
    data = np.random.default_rng(STUDENT_ID).random(ARRAY_SIZE, dtype=np.float64)
    print(f"Array size: {data.size:,}")
    print(f"Array dtype: {data.dtype}")
    print("Execution mode: serial (single thread)")

    start_ns = time.perf_counter_ns()
    total = np.sum(data, dtype=np.float64)
    std_dev = np.std(data, ddof=0, dtype=np.float64)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

    print(f"Sum: {total:.10f}")
    print(f"Standard deviation (population): {std_dev:.10f}")
    print(f"Wall-clock execution time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
