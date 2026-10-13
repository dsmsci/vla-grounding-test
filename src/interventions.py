"""Forward-output interventions and descriptive linear CKA, without training."""

import contextlib
import numpy as np
import torch
from .core import seed_for


def linear_cka(x, y):
    """Compute only debiased linear CKA; the ratio itself need not be unbiased.

    U-centering follows Google Research representation_similarity/Demo.ipynb
    (Apache-2.0; see NOTICE). Negative estimates are retained, not clipped.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("CKA needs two matrices with the same scene axis")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CKA inputs must be finite")
    n = len(x)
    if n < 4:
        return None
    grams = []
    for value in (x, y):
        value = value - value.mean(0)
        gram = value @ value.T
        scale = np.linalg.norm(gram)
        np.fill_diagonal(gram, 0)
        means = gram.sum(0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
        norm = np.linalg.norm(gram)
        if not np.isfinite(norm) or norm <= 1e-12 * scale:
            return None
        grams.append(gram / norm)
    return float((grams[0] * grams[1]).sum())


def matched_noise(delta, mask):
    """Draw a masked, norm-matched control without advancing the policy's RNG."""
    generator = torch.Generator(device=delta.device)
    generator.manual_seed(seed_for(torch.initial_seed(), "intervention_control"))
    noise = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
    noise = torch.where(mask, noise, 0.0)
    return noise * (delta.float().norm() / noise.float().norm().clamp_min(1e-12)).to(delta)


def patch(base, donor, mask, random=False):
    """Replace masked recipient activations by donor values; random=True is a control."""
    if base.shape != donor.shape:
        raise ValueError(f"Incompatible tensors: {base.shape} != {donor.shape}")
    mask = mask.to(base.device).bool().unsqueeze(-1).expand_as(base)
    donor = donor.to(base)
    delta = torch.where(mask, donor - base, 0.0)
    if random:
        delta = matched_noise(delta, mask)
    norm = torch.where(mask, base, 0.0).float().norm()
    rho = float(delta.float().norm() / norm) if norm > 0 else None
    return (base + delta if random else torch.where(mask, donor, base)), rho


def steer(base, direction, mask, alpha=1.0, random=False):
    """Add alpha times a frozen development direction; never substitute donor values."""
    if base.shape != direction.shape:
        raise ValueError(f"Incompatible tensors: {base.shape} != {direction.shape}")
    mask = mask.to(base.device).bool().unsqueeze(-1).expand_as(base)
    delta = torch.where(mask, direction.to(base), 0.0)
    if random:
        delta = matched_noise(delta, mask)
    norm = torch.where(mask, base, 0.0).float().norm()
    rho = float(abs(alpha) * delta.float().norm() / norm) if norm > 0 else None
    return base + alpha * delta, rho


def modules(model):
    """Expose functional interfaces and numbered residual block outputs, not depth percentiles."""
    return {
        "backbone": model.backbone,
        "context": model.action_head.vl_self_attention,
        **{f"dit_{i}": block for i, block in enumerate(model.action_head.model.transformer_blocks)},
    }


class Trace:
    """CPU activations, masks and sampled attention from one native forward."""

    def __init__(self):
        """Start an empty trace; values are populated by temporary PyTorch hooks."""
        self.values, self.meta, self.attention = {}, {}, {}
        self.patches, self.rho = 0, None


@contextlib.contextmanager
def hooks(model, capture=(), intervention=None, denoise=0, attention=False):
    """Capture CPU tensors and apply one patch OR steering edit at a flow-matching step."""
    trace, handles, counts = Trace(), [], {}
    wanted = set(capture) | ({intervention["site"]} if intervention else set())
    # Backbone metadata is also needed for context masks and attention plots.
    wanted.add("backbone")
    unknown = wanted - modules(model).keys()
    if unknown:
        raise ValueError(f"Unknown intervention sites: {sorted(unknown)}")

    def make_hook(site):
        """Build an output hook for one named functional interface."""

        def hook(module, args, output):
            """Capture the original activation before applying the requested edit."""
            ordinal = counts.get(site, 0)
            counts[site] = ordinal + 1
            if site.startswith("dit_") and ordinal != denoise:
                return output
            tensor = output["backbone_features"] if site == "backbone" else output
            if site == "backbone":
                for key in ("backbone_attention_mask", "image_mask"):
                    trace.meta[key] = output[key].detach().cpu().bool()
            if site in capture:
                trace.values[site] = tensor.detach().cpu().clone()
            if intervention and site == intervention["site"]:
                source_meta = intervention.get("meta")
                if source_meta and site in {"context", "backbone"}:
                    for key, value in source_meta.items():
                        if key in trace.meta and not torch.equal(value, trace.meta[key]):
                            raise ValueError(f"Unaligned {key}; no implicit padding/cropping allowed")
                if site.startswith("dit_"):
                    mask = torch.zeros(tensor.shape[:2], dtype=torch.bool, device=tensor.device)
                    mask[:, -model.action_head.config.action_horizon :] = True
                else:
                    mask = trace.meta["backbone_attention_mask"]
                if intervention["kind"] == "patch":
                    changed, trace.rho = patch(
                        tensor, intervention["source"], mask, intervention.get("random", False)
                    )
                elif intervention["kind"] == "steer":
                    changed, trace.rho = steer(
                        tensor,
                        intervention["source"],
                        mask,
                        intervention.get("alpha", 1.0),
                        intervention.get("random", False),
                    )
                else:
                    raise ValueError(f"Unknown intervention kind: {intervention['kind']}")
                trace.patches += 1
                if site == "backbone":
                    data = dict(output)
                    data["backbone_features"] = changed
                    return type(output)(data=data)
                return changed
            return output

        return hook

    def attention_hook(name):
        """Build a sampled cross-attention observer using native Q/K projections."""

        def pre(module, args, kwargs):
            """Compute diagnostic weights without changing native attention output."""
            ordinal = counts.get(name, 0)
            counts[name] = ordinal + 1
            if ordinal != denoise:
                return
            hidden = args[0] if args else kwargs["hidden_states"]
            context = kwargs.get("encoder_hidden_states")
            if context is None:
                return
            q, k = module.to_q(hidden), module.to_k(context)
            b, nq, width = q.shape
            q = q.reshape(b, nq, module.heads, width // module.heads).transpose(1, 2)
            k = k.reshape(b, -1, module.heads, width // module.heads).transpose(1, 2)
            scores = q[:, :, -model.action_head.config.action_horizon :].float() @ k.float().transpose(-2, -1)
            scores *= module.scale
            mask = kwargs.get("attention_mask")
            if mask is not None:
                if mask.ndim != 2 or mask.dtype != torch.bool:
                    raise ValueError("Review native attention masking before interpreting maps")
                scores.masked_fill_(~mask[:, None, None, :], float("-inf"))
            trace.attention[name] = scores.softmax(-1).mean((1, 2)).detach().cpu()

        return pre

    try:
        for site, module in modules(model).items():
            if site in wanted:
                handles.append(module.register_forward_hook(make_hook(site)))
        if attention:
            for i, block in enumerate(model.action_head.model.transformer_blocks):
                if i % 2 == 0:
                    handles.append(
                        block.attn1.register_forward_pre_hook(
                            attention_hook(f"cross_{i:02}"), with_kwargs=True
                        )
                    )
        yield trace
        if intervention and trace.patches != 1:
            raise RuntimeError(f"Expected exactly one intervention, observed {trace.patches}")
    finally:
        for handle in handles:
            handle.remove()
