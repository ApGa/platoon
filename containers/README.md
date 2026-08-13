# ABCI H200 training runtime

This directory replaces the unprovenanced Enroot SQSH from the source cluster
with a Singularity CE build rooted in an immutable OCI platform manifest. It is
specifically the build/runtime layer for Platoon's content-addressed OpenReward
Python environment; it does not bake a mutable Platoon checkout into the SIF.

## Locked inputs

`abci-training-runtime.lock` is the single shell-readable provenance manifest.
The important identities are:

- `lmsysorg/sglang:v0.5.10.post1-runtime`, linux/amd64 manifest
  `sha256:829404319e32cc15c1a1fa0e8a47b807f98154c6287079b0548d031f1af30882`;
- Python 3.12, PyTorch 2.9.1+cu129, torchvision 0.24.1+cu129, and
  torchaudio 2.9.1+cu129;
- Apex commit `6424da3b4faa6c8f062da4a48c424fff3f02d42d`;
- Transformer Engine 2.12.0 (tag commit
  `5671fd3675906cda1ade26c24a65d3dedd88eb89`) from three hash-checked PyPI
  artifacts, plus hash-locked CPython 3.12 wheels for its otherwise-unlocked
  ONNX/inspection dependencies;
- Megatron Bridge 0.4.0 and Megatron Core 0.17.0 from the Platoon uv lock;
- SM90 compilation (`TORCH_CUDA_ARCH_LIST=9.0+PTX`,
  `NVTE_CUDA_ARCHS=90`).

Transformer Engine userbuffers are deliberately not compiled yet. Enabling
them before validating the ABCI HPC-X MPI ABI would make the first parity run
depend on an untested MPI stack.

The Ubuntu packages installed in `%post` are named but not version-pinned
because no Ubuntu snapshot mirror was part of the recovered source manifest.
The published SIF SHA-256 therefore remains the final immutable identity. A
future rebuild is not claimed to be byte-for-byte identical until those Debian
artifacts are snapshot-pinned.

## Static validation

Run this on the login node; it performs no downloads or builds:

```bash
pbs-scripts/verify_training_runtime.sh --static
```

Both extension installers also support `--dry-run`, `--print-provenance`, and
`--verify-only VENV` modes. Their normal `VENV` interface is compatible with a
content-addressed environment builder: they validate the exact Python,
PyTorch, CUDA, and cuDNN ABI; serialize shared builds; publish cached wheels
atomically; and write a provenance marker into the venv.

## Build and publish

Build only on a compute node so OCI unpack and compiler scratch use local NVMe.
The checked-in one-node wrapper is ready for submission when the validation
ladder reaches the image-build gate:

```bash
qsub pbs-scripts/build-training-runtime.pbs
```

The helper refuses to overwrite an existing image. It builds with Singularity
CE fakeroot under `$PBS_LOCALDIR`, runs the definition's tests, checks the
provenance labels and Python/CUDA baseline, copies through a same-directory
partial file, verifies the copy, and atomically publishes the SIF plus a
`.sha256` sidecar.

After publication, validate the image on an H200 node:

```bash
pbs-scripts/verify_training_runtime.sh \
  --sif "$PLATOON_TRAINING_SIF" \
  --with-gpu
```

Once the content-addressed training venv has compiled both extensions, add
`--venv "$OPENREWARD_JOB_VENV"` to verify their provenance and importability.
