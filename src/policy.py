# Native inference sequence adapted from NVIDIA/Isaac-GR00T (Apache-2.0).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. See docs/NOTICE.md.
"""N1.7 adapter: native processor/decoder, inspectable eager PyTorch, frozen weights."""

import hashlib
import re
import time
import numpy as np
import torch
from .core import ROOT, PINS, read_json, seeded, atomic_json
from .interventions import hooks


class Policy:
    """Frozen native N1.7 policy with inspectable PyTorch output hooks."""

    def __init__(self, suite, cfg):
        """Load one suite at a time and reject incompatible hardware/checkpoint structure."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("CUDA GPU with native BF16 is required for this experiment")
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        path = ROOT / "checkpoints" / suite
        provenance = read_json(path / "provenance.json")
        if provenance["suite"] != suite:
            raise ValueError(f"Wrong checkpoint suite: expected {suite}, found {provenance['suite']}")
        if provenance["pins"] != PINS:
            raise ValueError("Checkpoint provenance mismatch: run download in this environment")
        for name, sha in provenance["metadata_sha256"].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Checkpoint metadata changed: {name}")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.native = Gr00tPolicy("LIBERO_PANDA", str(path), device="cuda:0")
        self.model, self.cfg = self.native.model, cfg
        self.model.requires_grad_(False)
        self.model.eval()
        torch.cuda.reset_peak_memory_stats()
        head = self.model.action_head.model
        if (
            getattr(head, "attend_text_every_n_blocks", None) != 2
            or not head.config.interleave_self_attention
        ):
            raise ValueError("Expected alternating non-image/image cross-attention with self-attention")
        if len(self.model.action_head.model.transformer_blocks) != 32:
            raise ValueError("Recheck functional sites: expected 32 DiT blocks")
        if self.model.action_head.config.num_inference_timesteps != 4:
            raise ValueError("Expected four inference timesteps")
        if self.model.action_head.config.action_horizon != 40:
            raise ValueError("Expected 40 padded action tokens")
        if self.native.modality_configs["action"].delta_indices != list(range(16)):
            raise ValueError("Expected 16 valid LIBERO actions")

    def forward(self, observation, noise, capture=(), intervention=None, attention=False):
        """Run the native processor/model/decoder with deterministic common action noise."""
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype
        from torch.nn.attention import SDPBackend, sdpa_kernel

        p = self.native
        p.check_observation(observation)
        start = time.monotonic()
        with seeded(noise), torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            step = p._to_vla_step_data(p._unbatch_observation(observation)[0])
            processed = p.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            inputs = _rec_to_dtype(p.collate_fn([processed]), torch.bfloat16)
            with hooks(self.model, capture, intervention, self.cfg.denoise_step, attention) as trace:
                predicted = self.model.get_action(**inputs)
            normalized = predicted["action_pred"].float().cpu().numpy()
            states = {k: v[None] for k, v in step.states.items()}
            decoded = p.processor.decode_action(normalized, p.embodiment_tag, states)
        actions = np.concatenate(
            [decoded[k][0] for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")], axis=-1
        )
        actions = actions.astype(np.float32)
        if normalized.shape != (1, 40, 132) or actions.shape != (16, 7):
            raise ValueError(f"Unexpected action shapes: padded={normalized.shape}, decoded={actions.shape}")
        adapted = actions.copy()
        adapted[:, -1] = -np.sign(2 * adapted[:, -1] - 1)
        if not np.isfinite(adapted).all():
            raise ValueError("Non-finite policy action")
        inner = inputs["inputs"]
        if capture or attention:
            ids = inner["input_ids"].detach().cpu()
            tokenizer = p.processor.collator.processor.tokenizer
            instruction = re.sub(r"[^\w\s]", "", step.text.lower())
            needle = tokenizer.encode(instruction, add_special_tokens=False)
            found = [
                i
                for i in range(ids.shape[1] - len(needle) + 1)
                if ids[0, i : i + len(needle)].tolist() == needle
            ]
            if len(found) != 1:
                raise ValueError("Cannot uniquely align instruction tokens; inspect tokenizer")
            text_mask = torch.zeros_like(ids, dtype=torch.bool)
            text_mask[:, found[0] : found[0] + len(needle)] = True
            trace.meta["text_mask"] = text_mask
            trace.tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
            trace.grid = inner["image_grid_thw"].detach().cpu().numpy()
            trace.merge = p.processor.collator.processor.image_processor.merge_size
        return dict(
            actions=adapted,
            decoded=actions,
            normalized=normalized,
            trace=trace,
            seconds=time.monotonic() - start,
            peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
        )

    def self_check(self, obs, noise, sites):
        """Verify that replacing an activation with itself preserves normalized actions."""
        clean = self.forward(obs, noise, capture=sites)
        results = {}
        for site in sites:
            trace = clean["trace"]
            patched = self.forward(
                obs,
                noise,
                intervention=dict(kind="patch", site=site, source=trace.values[site], meta=trace.meta),
            )
            error = float(np.max(np.abs(clean["normalized"] - patched["normalized"])))
            results[site] = error
            if not np.allclose(clean["normalized"], patched["normalized"], atol=1e-4, rtol=1e-4):
                raise RuntimeError(f"Self-patch failed at {site}, max error={error}")
        return results


def save_trace(path, trace):
    """Token maps and spatial token grids; these are not raw-image overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.mkdir(parents=True, exist_ok=True)
    grid = getattr(trace, "grid", None)
    atomic_json(
        path / "tokens.json",
        dict(
            tokens=getattr(trace, "tokens", None),
            grid=grid.tolist() if grid is not None else None,
            merge=getattr(trace, "merge", None),
            masks={k: v.tolist() for k, v in trace.meta.items()},
            shapes={k: list(v.shape) for k, v in trace.values.items()},
        ),
    )
    arrays = {k: v.float().numpy() for k, v in trace.attention.items()}
    if not arrays:
        return
    np.savez_compressed(path / "attention.npz", **arrays)
    matrix = np.concatenate(list(arrays.values()))
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set(
        xlabel="Joint-context token index",
        ylabel="DiT cross-attention block",
        yticks=range(len(arrays)),
        yticklabels=list(arrays),
    )
    fig.colorbar(im, ax=ax, label="Mean attention over action queries and heads")
    fig.tight_layout()
    fig.savefig(path / "attention.png", dpi=150)
    plt.close(fig)
    image_ids = np.flatnonzero(trace.meta["image_mask"][0].numpy())
    sizes = [(int(t), int(h) // trace.merge, int(w) // trace.merge) for t, h, w in trace.grid]
    if sum(np.prod(s) for s in sizes) != len(image_ids):
        raise ValueError("Vision-token grid mismatch; do not fabricate spatial maps")
    # Only image-attending blocks have nonzero image mass.
    image_rows = matrix[:, image_ids]
    rows = image_rows[image_rows.sum(1) > 1e-6]
    if len(rows):
        vector, offset = rows.mean(0), 0
        for camera, shape in enumerate(sizes):
            n = int(np.prod(shape))
            heat = vector[offset : offset + n].reshape(shape).mean(0)
            fig, ax = plt.subplots()
            ax.imshow(heat, cmap="magma")
            ax.set_title(f"Processed camera {camera}: image-token attention")
            fig.savefig(path / f"camera_{camera}.png", dpi=150)
            plt.close(fig)
            offset += n
