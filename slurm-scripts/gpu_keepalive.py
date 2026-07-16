"""Small periodic GPU workload for Slurm jobs with idle-GPU cancellation.

The workload runs a brief BF16 matrix-multiply burst on every visible CUDA
device. It is intentionally configurable from environment variables so the
Slurm script can tune the tradeoff between visibility to utilization sampling
and training overhead.
"""

import os
import sys
import time

TICK_INTERVAL = int(os.environ.get("KEEPALIVE_TICK_SEC", "5"))
MATMUL_DIM = int(os.environ.get("KEEPALIVE_MATMUL_DIM", "4096"))
MATMUL_REPS = int(os.environ.get("KEEPALIVE_MATMUL_REPS", "2000"))
MAX_RUNTIME_SEC = int(os.environ.get("KEEPALIVE_MAX_SEC", str(int(4.5 * 3600))))
START_DELAY = int(os.environ.get("KEEPALIVE_START_DELAY_SEC", "300"))
EXPECTED_GPUS = int(os.environ.get("KEEPALIVE_EXPECTED_GPUS", "0"))
MAX_CONSECUTIVE_ERRORS = int(os.environ.get("KEEPALIVE_MAX_CONSECUTIVE_ERRORS", "3"))
READY_DIR = os.environ.get("KEEPALIVE_READY_DIR")


def _publish_ready(device_count: int) -> None:
    """Atomically report that this Slurm task has initialized every GPU."""
    if not READY_DIR:
        return

    task_id = os.environ.get("SLURM_PROCID")
    if task_id is None:
        raise RuntimeError("KEEPALIVE_READY_DIR is set but SLURM_PROCID is unavailable")
    os.makedirs(READY_DIR, exist_ok=True)
    destination = os.path.join(READY_DIR, f"{task_id}.ready")
    temporary = os.path.join(READY_DIR, f".{task_id}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as marker:
        marker.write(f"pid={os.getpid()}\nhost={os.uname().nodename}\ntask={task_id}\ndevices={device_count}\n")
        marker.flush()
        os.fsync(marker.fileno())
    os.replace(temporary, destination)


def _run_burst(torch, streams, left_operands, right_operands, device_count: int) -> float:
    """Run and synchronize one utilization burst on every visible GPU."""
    tick_start = time.time()
    for device_idx in range(device_count):
        torch.cuda.set_device(device_idx)
        with torch.cuda.stream(streams[device_idx]):
            output = left_operands[device_idx]
            right = right_operands[device_idx]
            for _ in range(MATMUL_REPS):
                output = output @ right
            _ = output.sum()

    for device_idx in range(device_count):
        torch.cuda.set_device(device_idx)
        streams[device_idx].synchronize()
    return time.time() - tick_start


def main() -> int:
    if START_DELAY > 0:
        print(
            f"[gpu_keepalive] sleeping {START_DELAY}s before importing torch",
            flush=True,
        )
        time.sleep(START_DELAY)

    import torch

    device_count = torch.cuda.device_count()
    print(
        f"[gpu_keepalive] starting on {device_count} GPUs, tick={TICK_INTERVAL}s, max={MAX_RUNTIME_SEC}s",
        flush=True,
    )
    if device_count == 0:
        print("[gpu_keepalive] no CUDA devices visible; exiting", flush=True)
        return 1
    if EXPECTED_GPUS > 0 and device_count != EXPECTED_GPUS:
        print(
            f"[gpu_keepalive] expected {EXPECTED_GPUS} GPUs but found {device_count}; exiting",
            file=sys.stderr,
            flush=True,
        )
        return 1

    streams = []
    left_operands = []
    right_operands = []
    for device_idx in range(device_count):
        torch.cuda.set_device(device_idx)
        streams.append(torch.cuda.Stream(device=device_idx))
        left_operands.append(torch.randn(MATMUL_DIM, MATMUL_DIM, dtype=torch.bfloat16, device=f"cuda:{device_idx}"))
        right_operands.append(torch.randn(MATMUL_DIM, MATMUL_DIM, dtype=torch.bfloat16, device=f"cuda:{device_idx}"))
        torch.cuda.synchronize(device_idx)

    print(
        f"[gpu_keepalive] config: dim={MATMUL_DIM}, reps_per_tick={MATMUL_REPS}, interval={TICK_INTERVAL}s",
        flush=True,
    )

    # The readiness handshake guarantees useful GPU work, not merely a live
    # Python process.  A first-burst failure is fatal so the launcher cannot
    # proceed under the false belief that idle-GPU protection is active.
    try:
        burst_seconds = _run_burst(torch, streams, left_operands, right_operands, device_count)
    except Exception as exc:
        print(
            f"[gpu_keepalive] initial utilization burst failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"[gpu_keepalive] tick 1 ({device_count} GPUs touched, burst={burst_seconds:.2f}s)",
        flush=True,
    )
    _publish_ready(device_count)

    end_time = time.time() + MAX_RUNTIME_SEC
    tick = 1
    consecutive_errors = 0
    time.sleep(TICK_INTERVAL)
    while time.time() < end_time:
        tick += 1
        try:
            burst_seconds = _run_burst(torch, streams, left_operands, right_operands, device_count)
            consecutive_errors = 0
            if tick % 12 == 0:
                print(
                    f"[gpu_keepalive] tick {tick} ({device_count} GPUs touched, burst={burst_seconds:.2f}s)",
                    flush=True,
                )
        except Exception as exc:
            consecutive_errors += 1
            sys.stderr.write(
                f"[gpu_keepalive] error on tick {tick} "
                f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} consecutive): {exc}\n"
            )
            sys.stderr.flush()
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                sys.stderr.write(
                    "[gpu_keepalive] persistent CUDA failures; exiting so the launcher can replace this allocation\n"
                )
                sys.stderr.flush()
                return 1

        time.sleep(TICK_INTERVAL)

    print(f"[gpu_keepalive] reached MAX_RUNTIME_SEC ({MAX_RUNTIME_SEC}s); exiting", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
