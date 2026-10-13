"""Hydra configuration, deterministic seeds and small scientific state files."""

import contextlib
import dataclasses
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "groot": "d2b7e75b937e3ec9aa5dbc798f08b89692c49734",
    "libero": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "weights": "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
    "backbone": "9ce19a195e423419c349abfc86fd07178b230561",
}


def config(name="mvp", overrides=()):
    """Compose a named conf/*.yaml with Hydra's native key=value overrides."""
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    if cfg.development < 4 or cfg.test < 2:
        raise ValueError("Use at least four development and two test scenes.")
    if not cfg.sites or len(set(cfg.sites)) != len(cfg.sites):
        raise ValueError("Intervention sites must be a nonempty distinct list.")
    if not 1 <= cfg.execute <= 16 or cfg.max_steps < 1 or cfg.denoise_step not in range(4):
        raise ValueError("Invalid action horizon or flow-matching step.")
    if cfg.success_hold_steps < 1:
        raise ValueError("Invalid success criterion.")
    if cfg.seed < 0 or cfg.bootstrap < 1 or not cfg.alphas:
        raise ValueError("Invalid seed, bootstrap count or steering coefficients.")
    return cfg


def project_path(value):
    """Resolve paths relative to the repository, independently of the caller's cwd."""
    return (ROOT / value).resolve()


def digest(value):
    """Hash JSON-serializable scientific metadata without including credentials."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def seed_for(*parts):
    """Derive a deterministic scene/call stream from the single experiment seed."""
    return int(digest(parts)[:8], 16) % (2**31 - 1)


def scene_seed(cfg, family, split, index):
    """Keep resets paired and balance color identity across consecutive scene indices."""
    return (seed_for(cfg.seed, family, split, index) & ~1) | (index % 2)


def atomic_json(path, value):
    """Commit a complete decision/outcome file; incomplete writes are never reused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def read_json(path):
    """Read a protocol artifact; this is not a substitute for MLflow tracking."""
    return json.loads(Path(path).read_text())


def append_json(path, value):
    """Append a raw simulator/call trace for later physical auditing."""
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n")


@contextlib.contextmanager
def seeded(seed):
    """Use the same Python/NumPy/PyTorch seed for a paired forward, then restore RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            set_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclasses.dataclass(frozen=True)
class Cell:
    """One fixed rollout condition; there is one policy seed, not repeated seed trials."""

    family: str
    split: str
    index: int
    seed: int
    contrast: str = "text"
    condition: str = "baseline"
    site: str = ""
    alpha: float = 1.0

    @property
    def key(self):
        """Stable MLflow episode tag and raw trace directory name."""
        return (
            f"{self.family}_{self.split}_{self.contrast}_{self.condition}_{self.site or 'none'}"
            f"_scene{self.index:03}_seed{self.seed}_a{self.alpha:g}"
        )
