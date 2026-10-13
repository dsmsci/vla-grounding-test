"""Pinned upstream sources and local, revision-pinned inference checkpoints."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from .core import ROOT, PINS, atomic_json, read_json


URLS = {
    "groot": "https://github.com/NVIDIA/Isaac-GR00T.git",
    "libero": "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
}


def setup():
    """Fetch pinned upstream code; safely repeat an interrupted initial checkout."""
    for name, url in URLS.items():
        path = ROOT / "vendor" / name
        if not (path / ".git").exists():
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", str(path), "init"], check=True)
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "--verify", "HEAD"], capture_output=True)
        # Only an unborn vendor checkout may be rebuilt after interruption.
        # An existing HEAD is never reset: activate() rejects wrong/modified code.
        if head.returncode:
            for args in (
                ["fetch", "--depth", "1", url, PINS[name]],
                ["checkout", "--detach", "--force", "FETCH_HEAD"],
            ):
                subprocess.run(
                    ["git", "-C", str(path), *args],
                    check=True,
                    # Upstream LFS demo videos/data are not inference dependencies.
                    env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
                )
    activate()


def activate():
    """Select EGL, verify source commits and point LIBERO to its own assets."""
    for key, value in {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ.setdefault(key, value)
    for name in URLS:
        path = ROOT / "vendor" / name
        sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if sha != PINS[name]:
            raise ValueError(f"{name}: expected {PINS[name]}, found {sha}")
        subprocess.run(
            ["git", "-C", str(path), "diff", "--exit-code", "HEAD"], check=True, stdout=subprocess.DEVNULL
        )
        sys.path.insert(0, str(path))
    base = ROOT / "vendor/libero/libero/libero"
    config = ROOT / ".cache/libero"
    config.mkdir(parents=True, exist_ok=True)
    config.joinpath("config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(str(base / v))}"
            for k, v in {
                "benchmark_root": ".",
                "bddl_files": "bddl_files",
                "init_states": "init_files",
                "datasets": "datasets",
                "assets": "assets",
            }.items()
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config)


def download(families):
    """Download inference snapshots using the standard HF_TOKEN/HF_HOME environment."""
    from huggingface_hub import snapshot_download
    from .scenes import FAMILIES

    bb = Path(
        snapshot_download(
            "nvidia/Cosmos-Reason2-2B",
            revision=PINS["backbone"],
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    )
    # Native factory recognizes this name substring even for a local path.
    alias = ROOT / "checkpoints/nvidia/Cosmos-Reason2-2B"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        alias.unlink()
    if alias.exists():
        raise ValueError(f"Refusing to overwrite non-symlink {alias}")
    alias.symlink_to(bb, target_is_directory=True)
    for suite in sorted({FAMILIES[f]["suite"] for f in families}):
        cache = (
            Path(
                snapshot_download(
                    "nvidia/GR00T-N1.7-LIBERO",
                    revision=PINS["weights"],
                    allow_patterns=[f"{suite}/*.json", f"{suite}/*.safetensors"],
                )
            )
            / suite
        )
        destination = ROOT / "checkpoints" / suite
        destination.mkdir(parents=True, exist_ok=True)
        for source in cache.iterdir():
            if source.suffix not in {".json", ".safetensors"}:
                continue
            target = destination / source.name
            if target.is_symlink() or target.exists():
                target.unlink()
            if source.suffix == ".json":
                value = read_json(source)
                if source.name == "config.json":
                    value.update(model_name=str(alias), use_flash_attention=False, load_bf16=True)
                if source.name == "processor_config.json":
                    value["processor_kwargs"]["model_name"] = str(alias)
                atomic_json(target, value)
            else:
                target.symlink_to(source.resolve())
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.glob("*.json")
            if p.name != "provenance.json"
        }
        atomic_json(
            destination / "provenance.json",
            {
                "pins": PINS,
                "metadata_sha256": hashes,
                "suite": suite,
                "backbone_snapshot": str(bb),
                "suite_snapshot": str(cache),
            },
        )
        print(f"Prepared {destination}", flush=True)


def doctor():
    """Report the actual device, free/total VRAM, driver and installed package versions."""
    import torch

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "transformers",
                "diffusers",
                "mujoco",
                "robosuite",
                "hydra-core",
                "fire",
                "mlflow",
                "numpy",
                "scipy",
                "pandas",
                "opencv-python-headless",
                "huggingface-hub",
            )
        },
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        result.update(
            free_vram_gib=free / 2**30,
            total_vram_gib=total / 2**30,
            gpu=torch.cuda.get_device_name(0),
            vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
            bf16_native=torch.cuda.is_bf16_supported(including_emulation=False),
        )
        result["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
    return result
