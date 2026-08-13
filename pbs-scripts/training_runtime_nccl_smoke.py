#!/usr/bin/env python3
"""Distributed CUDA/NCCL validation for the immutable ABCI training SIF."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import os
import socket
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    return parser.parse_args()


def fixed_bytes(value: str, length: int, torch):
    encoded = value.encode("utf-8")
    if len(encoded) >= length:
        raise AssertionError(f"value is too long for a {length}-byte field: {value!r}")
    return torch.tensor(
        list(encoded) + [0] * (length - len(encoded)),
        dtype=torch.uint8,
        device="cuda",
    )


def decode_fixed_bytes(tensor) -> str:
    payload = bytes(tensor.cpu().tolist())
    return payload.split(b"\0", 1)[0].decode("utf-8")


def main() -> int:
    args = parse_args()
    if args.world_size != 16:
        raise AssertionError(f"Gate 1 requires 16 ranks, got {args.world_size}")
    if not 0 <= args.local_rank < 8:
        raise AssertionError(
            f"local rank is outside one full H200 node: {args.local_rank}"
        )

    # --no-home/--containall deliberately hide the submitter's home. Runtime
    # libraries such as FlashInfer and Triton still require a writable JIT cache,
    # so the launcher supplies private paths in the container's ephemeral /tmp.
    for variable in (
        "HOME",
        "XDG_CACHE_HOME",
        "FLASHINFER_WORKSPACE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
    ):
        path = os.environ.get(variable)
        if not path or not path.startswith("/tmp/"):
            raise AssertionError(
                f"rank {args.rank}: {variable} is not a private /tmp path: {path!r}"
            )
        Path(path).mkdir(parents=True, exist_ok=True)

    # Import the modules needed by the runtime before initializing the process
    # group. This catches incomplete SIFs while leaving Apex/Transformer Engine
    # to the later content-addressed venv gate.
    imported = []
    for module_name in (
        "torch",
        "torchvision",
        "torchaudio",
        "sglang",
        "flashinfer",
        "triton",
    ):
        importlib.import_module(module_name)
        imported.append(module_name)

    import torch
    import torch.distributed as dist

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise AssertionError(
            f"rank {args.rank}: the UUID-isolated GPU must be container "
            f"ordinal 0, got CUDA_VISIBLE_DEVICES={visible!r}"
        )
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if nvidia_visible != args.expected_gpu_uuid:
        raise AssertionError(
            f"rank {args.rank}: NVIDIA_VISIBLE_DEVICES={nvidia_visible!r}, "
            f"expected {args.expected_gpu_uuid!r}"
        )
    if not torch.cuda.is_available():
        raise AssertionError(f"rank {args.rank}: CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise AssertionError(
            f"rank {args.rank}: GPU isolation failed; "
            f"visible device count={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if "H200" not in properties.name:
        raise AssertionError(
            f"rank {args.rank}: expected an H200, got {properties.name!r}"
        )
    if properties.major != 9:
        raise AssertionError(
            f"rank {args.rank}: expected Hopper compute capability, "
            f"got {properties.major}.{properties.minor}"
        )

    # Exercise kernels and HBM rather than merely asking the CUDA runtime for a
    # device count. A deterministic result makes silent launch/driver failures
    # visible without allocating enough memory to disturb neighboring ranks.
    left = torch.full(
        (1024, 1024),
        float(args.rank + 1),
        dtype=torch.float32,
        device="cuda",
    )
    right = torch.eye(1024, dtype=torch.float32, device="cuda")
    product = torch.mm(left, right)
    torch.cuda.synchronize()
    checksum = float(product.sum().item())
    expected_checksum = float((args.rank + 1) * 1024 * 1024)
    if not math.isclose(checksum, expected_checksum, rel_tol=0.0, abs_tol=1.0):
        raise AssertionError(
            f"rank {args.rank}: CUDA matmul checksum {checksum} "
            f"!= {expected_checksum}"
        )

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=5),
        device_id=torch.device("cuda", 0),
    )
    try:
        rank_sum = torch.tensor(float(args.rank + 1), device="cuda")
        dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM)
        expected_sum = args.world_size * (args.world_size + 1) / 2
        if rank_sum.item() != expected_sum:
            raise AssertionError(
                f"rank {args.rank}: NCCL all-reduce returned "
                f"{rank_sum.item()}, expected {expected_sum}"
            )

        # Prove the collective crossed exactly two hosts with eight unique local
        # ranks on each. Host names travel over NCCL as fixed-size GPU tensors.
        hostname = socket.gethostname()
        host_tensor = fixed_bytes(hostname, 128, torch)
        rank_tensor = torch.tensor(
            [args.rank, args.local_rank],
            dtype=torch.int64,
            device="cuda",
        )
        gathered_hosts = [
            torch.empty_like(host_tensor) for _ in range(args.world_size)
        ]
        gathered_ranks = [
            torch.empty_like(rank_tensor) for _ in range(args.world_size)
        ]
        dist.all_gather(gathered_hosts, host_tensor)
        dist.all_gather(gathered_ranks, rank_tensor)
        dist.barrier()

        if args.rank == 0:
            hosts = [decode_fixed_bytes(item) for item in gathered_hosts]
            ranks = [tuple(item.cpu().tolist()) for item in gathered_ranks]
            if sorted(global_rank for global_rank, _ in ranks) != list(
                range(args.world_size)
            ):
                raise AssertionError(f"unexpected global-rank map: {ranks}")
            per_host: dict[str, list[int]] = defaultdict(list)
            for host, (_, local_rank) in zip(hosts, ranks, strict=True):
                per_host[host].append(local_rank)
            if len(per_host) != 2:
                raise AssertionError(
                    f"expected two NCCL hosts, got {dict(per_host)}"
                )
            for host, local_ranks in per_host.items():
                if sorted(local_ranks) != list(range(8)):
                    raise AssertionError(
                        f"host {host} has invalid local-rank map: {local_ranks}"
                    )
            print(
                f"runtime_imports=ok modules={','.join(imported)}",
                flush=True,
            )
            print(f"rank_topology=ok hosts={dict(per_host)}", flush=True)
            print(
                "cuda_kernel=ok "
                f"device={properties.name!r} "
                f"capability={properties.major}.{properties.minor}",
                flush=True,
            )
            print(
                f"nccl_collective=ok backend={dist.get_backend()} "
                f"world_size={args.world_size} "
                f"all_reduce_sum={rank_sum.item():.1f}",
                flush=True,
            )

        # Each rank emits an auditable mapping. The short hash makes duplicate
        # physical-device mappings detectable without depending on pynvml.
        device_identity = f"{hostname}:{args.local_rank}:{properties.name}"
        print(
            f"rank_passed rank={args.rank} local_rank={args.local_rank} "
            f"host={hostname} visible={visible} "
            f"nvidia_visible={nvidia_visible} "
            f"device_count={torch.cuda.device_count()} "
            "device_key="
            f"{hashlib.sha256(device_identity.encode()).hexdigest()[:16]} "
            f"marker={args.marker}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        print(f"rank_failed error={exc!r}", file=sys.stderr, flush=True)
        raise
