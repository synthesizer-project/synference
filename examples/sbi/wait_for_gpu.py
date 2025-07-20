"""A Python wrapper to delay a command until a GPU is available.

This script checks the availability of GPUs based on VRAM and utilization thresholds,
and waits until a GPU is free before executing a specified command. It uses the NVIDIA Management Library (NVML) to monitor GPU status
so will only work on systems with NVIDIA GPUs and the appropriate drivers installed.

Usage:
    python wait_for_gpu.py --max-vram 50 --max-util 10 -- python my_script.py --arg1 value1
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )
except ImportError:
    print("Error: The 'pynvml' library is not installed.", file=sys.stderr)
    print("Please install it using: pip install pynvml", file=sys.stderr)
    sys.exit(1)


def find_free_gpu(max_vram: float, max_util: float) -> Optional[int]:
    """
    Finds an available GPU based on VRAM and utilization thresholds.

    Args:
        max_vram: The maximum VRAM usage percentage.
        max_util: The maximum GPU utilization percentage.

    Returns:
        The integer ID of a free GPU, or None if none are free.
    """
    try:
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_rates = nvmlDeviceGetUtilizationRates(handle)

            vram_percent = (mem_info.used / mem_info.total) * 100
            gpu_util = util_rates.gpu

            if vram_percent < max_vram and gpu_util < max_util:
                # Found a free GPU, return its ID
                return i
    except NVMLError as e:
        print(f"NVML Error while checking GPUs: {e}", file=sys.stderr)
    return None

def get_gpu_statuses(max_vram: float, max_util: float) -> str:
    """Gets a string summary of all GPU statuses."""
    statuses: List[str] = []
    try:
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_rates = nvmlDeviceGetUtilizationRates(handle)
            vram_percent = (mem_info.used / mem_info.total) * 100
            gpu_util = util_rates.gpu

            vram_ok = "✅" if vram_percent < max_vram else "❌"
            util_ok = "✅" if gpu_util < max_util else "❌"

            statuses.append(
                f"GPU {i}: [VRAM: {vram_percent:5.1f}% {vram_ok} | "
                f"Util: {gpu_util:3.0f}% {util_ok}]"
            )
    except NVMLError:
        return "Could not retrieve GPU status from NVML."
    return " | ".join(statuses)


def main() -> None:
    """Main function to parse arguments and wait for a free GPU."""
    parser = argparse.ArgumentParser(
        description="A Python wrapper to delay a command until a GPU is available.",
        epilog=(
            "Example: python wait_for_gpu.py --max-vram 50 --max-util 10 "
            "-- python my_script.py --arg1 value1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-vram",
        type=float,
        default=20.0,
        help="Maximum VRAM usage %% to consider a GPU free (default: 20.0).",
    )
    parser.add_argument(
        "--max-util",
        type=float,
        default=10.0,
        help="Maximum GPU utilization %% to consider a GPU free (default: 10.0).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command to execute once a GPU is free.",
    )
    args = parser.parse_args()

    if not args.command:
        parser.error("You must specify a command to run.")

    try:
        nvmlInit()
        while True:
            free_gpu_id = find_free_gpu(args.max_vram, args.max_util)

            if free_gpu_id is not None:
                # Clear the status line before printing the success message
                sys.stdout.write("\r" + " " * 120 + "\r")
                sys.stdout.flush()
                print(f"✅ GPU {free_gpu_id} is available. Starting command.")
                break  # Exit the waiting loop

            status_summary = get_gpu_statuses(args.max_vram, args.max_util)
            sys.stdout.write(f"\r🕒 Waiting... {status_summary}")
            sys.stdout.flush()
            time.sleep(args.interval)

        # Set environment variable to make only the free GPU visible to the process
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)

        try:
            # Execute the provided command
            result = subprocess.run(args.command, check=False, env=env)
            if result.returncode == 0:
                print("\n✅ Command finished successfully.")
            else:
                print(
                    f"\n❌ Command failed with exit code {result.returncode}.",
                    file=sys.stderr,
                )
            sys.exit(result.returncode)
        except FileNotFoundError:
            print(
                f"\n❌ Command not found: '{args.command[0]}'. "
                "Check the command and your system's PATH.",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An error occurred while running the command: {e}", file=sys.stderr)
            sys.exit(1)

    except NVMLError as e:
        print(f"NVML Error: {e}", file=sys.stderr)
        print("Please ensure NVIDIA drivers are installed correctly.", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass # Can happen if nvmlInit() failed


if __name__ == "__main__":
    main()