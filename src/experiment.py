"""Development-only choices, fixed-site patching and frozen held-out evaluation."""

import hashlib
import logging
from itertools import product

import mlflow
import numpy as np
import torch
from PIL import Image

from .core import Cell, atomic_json, read_json, scene_seed, seed_for
from .interventions import linear_cka
from .policy import save_trace

LOGGER = logging.getLogger(__name__)


class Experiment:
    """Run the scientific matrix; CKA never selects or removes intervention sites."""

    def __init__(self, engine):
        """Attach one suite policy/environment and the resolved Hydra configuration."""
        self.e, self.cfg = engine, engine.cfg
        self.family, self.out = engine.family, engine.out
        self.path = self.out / "decisions" / self.family
        self.path.mkdir(parents=True, exist_ok=True)

    def run_cells(self, split, contrast, conditions, site="", alpha=1.0, direction=None, count=None):
        """Evaluate every requested condition on paired scenes with the single policy seed."""
        n = getattr(self.cfg, split) if count is None else count
        cells = [
            Cell(self.family, split, i, self.cfg.seed, contrast, condition, site, alpha)
            for i, condition in product(range(n), conditions)
        ]
        return self.e.cells(cells, direction)

    def diagnose(self):
        """Measure debiased CKA and sampled attention at all predeclared sites."""
        path = self.out / "diagnostics" / self.family
        path.mkdir(parents=True, exist_ok=True)
        if (path / "done.json").exists():
            return
        scene, policy, sites = self.e.scene, self.e.policy, tuple(self.cfg.sites)
        scene.reset(scene_seed(self.cfg, self.family, "development", 0))
        check = policy.self_check(scene.observation(), seed_for(self.cfg.seed, 0), sites)
        atomic_json(path / "self_check.json", check)
        pools = {}
        for i in range(self.cfg.development):
            scene.reset(scene_seed(self.cfg, self.family, "development", i))
            text = (i // 2) % 2
            variants = [("base", text, 0), ("text", 1 - text, 0)]
            if self.family.endswith("color"):
                variants.append(("visual", text, 1))
            for label, t, v in variants:
                observation = scene.observation(t, v)
                if i < 2:
                    sample = path / f"scene_{i}"
                    sample.mkdir(exist_ok=True)
                    for camera, frames in observation["video"].items():
                        Image.fromarray(frames[0, 0]).save(sample / f"{label}_{camera}.png")
                    atomic_json(sample / f"{label}_input.json", dict(input_bits=[t, v], **scene.snapshot()))
                output = policy.forward(
                    observation,
                    seed_for(self.cfg.seed, i),
                    capture=sites,
                    attention=i < 2 and label == "base",
                )
                trace = output["trace"]
                if i < 2 and label == "base":
                    save_trace(path / f"scene_{i}", trace)
                for site, value in trace.values.items():
                    if site.startswith("dit_"):
                        mask = torch.zeros(value.shape[:2], dtype=torch.bool)
                        mask[:, -policy.model.action_head.config.action_horizon :] = True
                        masks = {"action": mask}
                    else:
                        masks = {"text": trace.meta["text_mask"], "image": trace.meta["image_mask"]}
                    for modality, mask in masks.items():
                        pools.setdefault(f"{site}/{modality}/{label}", []).append(
                            value.float()[mask].mean(0).numpy()
                        )
            LOGGER.info("diagnostic %s scene %d/%d", self.family, i + 1, self.cfg.development)
        arrays = {key: np.stack(values) for key, values in pools.items()}
        np.savez_compressed(path / "pooled.npz", **arrays)
        comparisons = [
            (key, key.removesuffix("base") + contrast)
            for key in arrays
            if key.endswith("/base")
            for contrast in ("text", "visual")
            if key.removesuffix("base") + contrast in arrays
        ]
        comparisons += [
            (f"{site}/text/base", f"{site}/image/base")
            for site in ("backbone", "context")
            if f"{site}/text/base" in arrays
        ]
        cka, rng = {}, np.random.default_rng(self.cfg.seed)
        for left, right in comparisons:
            a, b = arrays[left], arrays[right]
            score = linear_cka(a, b)
            cka[f"{left} vs {right}"] = dict(
                estimator="debiased_linear_hsic_u",
                cka=score,
                shuffled=linear_cka(a, b[rng.permutation(len(b))]),
                n=len(a),
            )
            if score is not None:
                mlflow.log_metric(f"cka/{self.family}/{left}_vs_{right}", score)
        atomic_json(path / "cka.json", cka)
        cosine = {}
        for key in arrays:
            prefix = key.removesuffix("base")
            if key.endswith("/base") and prefix + "visual" in arrays:
                dt, dv = arrays[prefix + "text"] - arrays[key], arrays[prefix + "visual"] - arrays[key]
                denominator = np.linalg.norm(dt, axis=1) * np.linalg.norm(dv, axis=1)
                values = [float(np.dot(a, b) / d) if d > 0 else None for a, b, d in zip(dt, dv, denominator)]
                valid = [value for value in values if value is not None]
                cosine[prefix] = dict(per_scene=values, mean=float(np.mean(valid)) if valid else None)
        atomic_json(path / "cosine.json", cosine)
        atomic_json(path / "done.json", dict(self_check=check, sites=sites, n=self.cfg.development))

    def competence(self, contrast):
        """Describe baseline completion and factor choice without gating interventions."""
        path = self.path / f"{contrast}_competence.json"
        if path.exists():
            return read_json(path)
        rows = self.run_cells("development", contrast, ["baseline", "alternate"])
        rates = {
            c: float(np.mean([r["success"] for r in rows if r["condition"] == c]))
            for c in ("baseline", "alternate")
        }
        result = dict(
            rates=rates,
            factor_rates={
                c: {
                    m: float(np.mean([r[m] for r in rows if r["condition"] == c]))
                    for m in ("T_correct", "G_correct", "R", "choice0", "choice1")
                }
                for c in ("baseline", "alternate")
            },
            role="diagnostic_only",
            denominator=self.cfg.development,
        )
        atomic_json(path, result)
        LOGGER.info("competence %s/%s: %s", self.family, contrast, result)
        return result

    def prepare_patch(self, contrast):
        """Compare bidirectional factor-choice effects, retaining completion as diagnostics."""
        self.competence(contrast)
        path = self.path / f"{contrast}_patch.json"
        if path.exists():
            return read_json(path)
        base = self.run_cells("development", contrast, ["baseline", "alternate"])
        scores, candidates = {}, {}
        for site in self.cfg.sites:
            rows = self.run_cells("development", contrast, ["patch_to_alt", "patch_to_base"], site=site)
            candidates[site] = rows
            forward = np.mean([r["choice1"] for r in rows if r["condition"] == "patch_to_alt"])
            forward -= np.mean([r["choice1"] for r in base if r["condition"] == "baseline"])
            reverse = np.mean([r["choice0"] for r in rows if r["condition"] == "patch_to_base"])
            reverse -= np.mean([r["choice0"] for r in base if r["condition"] == "alternate"])
            scores[site] = float((forward + reverse) / 2)
        site = min(scores, key=lambda value: (-scores[value], value))
        controls = self.run_cells("development", contrast, ["patch_same", "patch_random"], site=site)
        rows = base + candidates[site] + controls

        def mean(condition, metric):
            """Average one complete selected-site condition."""
            return float(np.mean([r[metric] for r in rows if r["condition"] == condition]))

        forward = mean("patch_to_alt", "choice1") - mean("baseline", "choice1")
        reverse = mean("patch_to_base", "choice0") - mean("alternate", "choice0")
        drop = mean("baseline", "preserve") - mean("patch_to_alt", "preserve")
        gains = {
            c: mean("patch_to_alt", "choice1") - mean(c, "choice1") for c in ("patch_same", "patch_random")
        }
        self_row = self.run_cells("development", contrast, ["patch_self"], site=site, count=1)[0]
        reference = next(r for r in base if r["condition"] == "baseline" and r["index"] == 0)
        if any(self_row[m] != reference[m] for m in ("T_correct", "G_correct", "R", "steps")):
            raise RuntimeError(f"Self-patch changed rollout outcomes at {site}")
        result = dict(
            site=site,
            metric="factor_choice",
            development_scores=scores,
            forward=forward,
            reverse=reverse,
            preservation_drop=drop,
            preservation_within_tolerance=drop <= self.cfg.preservation_tolerance,
            controls=gains,
            positive=forward > 0
            and reverse > 0
            and min(gains.values()) > 0,
        )
        atomic_json(path, result)
        LOGGER.info("selected development patch %s/%s: %s", self.family, contrast, result)
        return result

    def direction(self, site):
        """Average token-aligned text differences over all development scenes, without success filtering."""
        path, metadata = self.path / "direction.pt", self.path / "direction.json"
        if path.exists() and metadata.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != read_json(metadata)["sha256"]:
                raise ValueError("Development direction changed; use a new output.")
            return torch.load(path, map_location="cpu", weights_only=True)
        deltas, same, sources, reference = [], [], [], None
        scene, policy = self.e.scene, self.e.policy
        for i in range(self.cfg.development):
            scene.reset(scene_seed(self.cfg, self.family, "development", i))
            traces = [
                policy.forward(
                    scene.observation(t, 0, paraphrase=p), seed_for(self.cfg.seed, 0), capture=[site]
                )["trace"]
                for t, p in ((0, False), (1, False), (0, True))
            ]
            reference = reference or traces[0]
            for trace in traces:
                if trace.values[site].shape != reference.values[site].shape:
                    raise ValueError("Steering source shape mismatch")
                if site in {"context", "backbone"}:
                    for key in ("image_mask", "backbone_attention_mask"):
                        if not torch.equal(trace.meta[key], reference.meta[key]):
                            raise ValueError("Steering source mask mismatch")
            deltas.append(traces[1].values[site].float() - traces[0].values[site].float())
            same.append(traces[2].values[site].float() - traces[0].values[site].float())
            target = scene.f["targets"][1 ^ scene.parity]
            sources.append(dict(index=i, target=target, position=scene.position(target).tolist()))
        result = dict(
            delta=torch.stack(deltas).mean(0),
            same=torch.stack(same).mean(0),
            meta=reference.meta,
            sources=sources,
            source_id=f"development/{self.family}/{site}",
        )
        temporary = path.with_suffix(".tmp")
        torch.save(result, temporary)
        temporary.replace(path)
        atomic_json(
            metadata, dict(site=site, sources=sources, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
        return result

    def prepare_steering(self, patches):
        """Tune steering on factor choice after a bidirectional patch beats its controls."""
        if self.family[0] != "T":
            return None
        if not patches or not patches["positive"]:
            return dict(
                site=None, alpha=None,
                reason="Development factor-choice patch did not improve both directions and beat controls",
            )
        path = self.path / "steering.json"
        if path.exists():
            return read_json(path)
        site = patches["site"]
        direction = self.direction(site)
        base = self.run_cells("development", "text", ["baseline"])
        preservation = float(np.mean([r["preserve"] for r in base]))
        scores, drops = {}, {}
        for alpha in self.cfg.alphas:
            rows = self.run_cells(
                "development", "text", ["steer"], site=site, alpha=alpha, direction=direction
            )
            scores[alpha] = float(np.mean([r["choice1"] for r in rows]))
            drops[alpha] = preservation - float(np.mean([r["preserve"] for r in rows]))
        alpha = min(scores, key=lambda a: (-scores[a], a))
        result = dict(
            site=site,
            metric="choice1",
            alpha=alpha,
            scores=scores,
            preservation_drops=drops,
            preservation_within_tolerance={a: d <= self.cfg.preservation_tolerance for a, d in drops.items()},
            reason=None,
        )
        atomic_json(path, result)
        return result

    def patch(self):
        """Run development diagnostics and patch matrices; reuse completed MLflow episodes."""
        self.diagnose()
        return {c: self.prepare_patch(c) for c in self.contrasts}

    @property
    def contrasts(self):
        """Text-only for native tasks; text and visual for color tasks."""
        return ("text", "visual") if self.family.endswith("color") else ("text",)

    def steer(self):
        """Complete the separate steering stage and freeze all choices before test."""
        frozen = self.path / "frozen.json"
        if frozen.exists():
            return read_json(frozen)
        patches = {}
        for contrast in self.contrasts:
            path = self.path / f"{contrast}_patch.json"
            if not path.exists():
                raise ValueError("Finish the patch stage before steering/freezing.")
            patches[contrast] = read_json(path)
        steering = self.prepare_steering(patches["text"])
        result = dict(
            patches=patches,
            steering=steering,
            mixed=self.family.endswith("color"),
            seed=self.cfg.seed,
            sites=list(self.cfg.sites),
            development_scenes=self.cfg.development,
            test_scenes=self.cfg.test,
        )
        if steering and steering["alpha"] is not None:
            result["direction_sha256"] = hashlib.sha256((self.path / "direction.pt").read_bytes()).hexdigest()
        atomic_json(frozen, result)
        LOGGER.info("frozen %s: %s", self.family, result)
        return result

    def test(self):
        """Evaluate the frozen development-selected patch and steering without new selection."""
        path = self.path / "frozen.json"
        if not path.exists():
            raise ValueError("Finish develop (or patch then steer) before test.")
        frozen = read_json(path)
        for contrast, gate in frozen["patches"].items():
            if gate is None:
                continue
            self.run_cells("test", contrast, ["baseline", "alternate"])
            self.run_cells(
                "test",
                contrast,
                ["patch_to_alt", "patch_to_base", "patch_same", "patch_random"],
                site=gate["site"],
            )
        if frozen["mixed"]:
            for prefix, site in (("mixed_", ""), ("mixedpatch_", "context")):
                self.run_cells(
                    "test", "mixed", [prefix + f"{t}{v}" for t, v in product(range(2), repeat=2)], site=site
                )
        steering = frozen["steering"]
        if steering and steering["alpha"] is not None:
            direction_path = self.path / "direction.pt"
            if hashlib.sha256(direction_path.read_bytes()).hexdigest() != frozen["direction_sha256"]:
                raise ValueError("Frozen steering direction changed.")
            direction = torch.load(direction_path, map_location="cpu", weights_only=True)
            self.run_cells(
                "test",
                "text",
                ["steer", "steer_minus", "steer_same", "steer_random", "steer_zero"],
                site=steering["site"],
                alpha=steering["alpha"],
                direction=direction,
            )

    def run(self, stage):
        """Dispatch a scientific stage; setup/report/UI are handled by the launcher."""
        if stage == "smoke":
            self.e.scene.reset(scene_seed(self.cfg, self.family, "development", 0))
            check = self.e.policy.self_check(
                self.e.scene.observation(), seed_for(self.cfg.seed, 0), self.cfg.sites
            )
            atomic_json(self.out / "diagnostics" / self.family / "smoke_self_check.json", check)
            self.run_cells("development", "text", ["baseline", "alternate"], count=1)
            for site in ("context", "dit_0"):
                if site in self.cfg.sites:
                    self.run_cells("development", "text", ["patch_to_alt"], site=site, count=1)
        elif stage == "diagnose":
            self.diagnose()
        elif stage == "baseline":
            for contrast in self.contrasts:
                self.competence(contrast)
        elif stage == "patch":
            self.patch()
        elif stage == "steer":
            self.steer()
        elif stage == "develop":
            self.patch()
            self.steer()
        elif stage == "test":
            self.test()
        else:
            raise ValueError(f"Unknown experiment stage: {stage}")
