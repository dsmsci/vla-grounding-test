"""Closed-loop execution. Completed episodes are atomic; interrupted ones restart from reset."""

import dataclasses
import logging

import mlflow
import time
import numpy as np
import torch
from .core import read_json, atomic_json, append_json, scene_seed, seed_for, digest, project_path
from .scenes import Scene, bits
from .policy import save_trace
from .tracking import completed_results, episode_run, log_outcome

LOGGER = logging.getLogger(__name__)


def inputs_for(cell):
    """Describe recipient/donor inputs and the independent patch control."""
    condition = cell.condition
    recipient = bits(cell.contrast, int(condition in {"alternate", "patch_to_base"}))
    donor, paraphrase, mode = None, False, "patch"
    if condition.startswith("mixedpatch_"):
        donor = tuple(map(int, condition.rsplit("_", 1)[1]))
    elif condition.startswith("mixed_"):
        recipient = tuple(map(int, condition.rsplit("_", 1)[1]))
    elif condition in {"patch_to_alt", "patch_to_base", "patch_random", "patch_self", "patch_same"}:
        donor = bits(cell.contrast, int(condition in {"patch_to_alt", "patch_random"}))
        paraphrase = condition == "patch_same"
        mode = "random" if condition == "patch_random" else "patch"
    return recipient, donor, paraphrase, mode


class Rollout:
    """Closed-loop execution with native environment success checks and MLflow episodes."""

    def __init__(self, cfg, policy, family):
        """Create one environment and index completed episodes through MLflow."""
        self.cfg, self.policy = cfg, policy
        self.out, self.videos = project_path(cfg.output), project_path(cfg.videos)
        self.scene = Scene(family, cfg.success_hold_steps)
        self.family = family
        self.completed = {row["key"]: row for row in completed_results()}

    def episode(self, cell, direction=None):
        """Reuse a completed MLflow episode or resume its incomplete attempt from reset."""
        if cell.key in self.completed:
            return self.completed[cell.key]
        with episode_run(cell) as (run_id, attempt):
            result = self.execute(cell, direction, attempt)
            log_outcome(result)
            result["mlflow_run_id"] = run_id
        self.completed[cell.key] = result
        return result

    def execute(self, cell, direction=None, attempt=1):
        """Execute one attempt; failed attempts remain visible and never receive success labels."""
        cfg, scene = self.cfg, self.scene
        folder = self.out / "episodes" / cell.key / f"attempt_{attempt}"
        folder.mkdir(parents=True, exist_ok=True)
        scene_id = scene_seed(cfg, cell.family, cell.split, cell.index)
        video_path = self.videos / f"{cell.key}.mp4"
        writer = None
        grasp_eef = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        recipient, donor, paraphrase, mode = inputs_for(cell)
        chunk, phase_calls, intervention_calls, intervention_steps, inference_seconds, forwards = (
            0,
            0,
            0,
            0,
            0.0,
            0,
        )
        first_intervention, last_intervention = None, None
        eef_path_m, eef_max_step_m = 0.0, 0.0
        LOGGER.info("episode start %s attempt=%d scene=%d", cell.key, attempt, scene_id)
        try:
            initial = scene.reset(scene_id)
            # All conditions and noise seeds must start from the very same physical scene.
            fingerprint = digest({k: initial[k] for k in ("state", "poses", "goal_positions")})
            reference = self.out / "scenes" / f"{cell.family}_{scene_id}.json"
            if reference.exists():
                if read_json(reference)["fingerprint"] != fingerprint:
                    raise RuntimeError("Unpaired reset: physical state differs from this scene's reference")
            else:
                atomic_json(reference, dict(fingerprint=fingerprint, scene_seed=scene_id, **initial))
            atomic_json(
                folder / "initial.json",
                dict(
                    scene_seed=scene_id,
                    fingerprint=fingerprint,
                    input_bits=recipient,
                    color_parity=scene.parity,
                    **initial,
                ),
            )
            scene.set_visual(recipient[1])
            if cfg.save_videos:
                import imageio.v2 as imageio

                self.videos.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(video_path, fps=20, macro_block_size=16)
                writer.append_data(scene.frame())
            while scene.step_number < cfg.max_steps:
                obs = scene.observation(*recipient)
                noise = seed_for(cell.seed, chunk)
                intervention, source_id, source_trace, shadow_record = None, None, None, None
                active = scene.eligible
                if active:
                    phase_calls += 1
                if donor is not None and active:
                    source_obs = scene.observation(*donor, paraphrase=paraphrase)
                    shadow = self.policy.forward(source_obs, noise, capture=(cell.site,))
                    forwards += 1
                    inference_seconds += shadow["seconds"]
                    source_trace = shadow["trace"]
                    source_id = f"{cell.key}/shadow/{chunk}"
                    shadow_record = dict(
                        instruction=next(iter(source_obs["language"].values()))[0][0],
                        noise_seed=noise,
                        normalized=shadow["normalized"][0, : len(shadow["actions"]), :7].tolist(),
                        decoded=shadow["decoded"].tolist(),
                        actions=shadow["actions"].tolist(),
                        seconds=shadow["seconds"],
                    )
                    intervention = dict(
                        site=cell.site,
                        source=source_trace.values[cell.site],
                        meta=source_trace.meta,
                        kind="patch",
                        random=mode == "random",
                    )
                    if cfg.save_images:
                        self.save_images(folder, source_obs, f"{chunk:03}_source")
                if cell.condition.startswith("steer") and active:
                    if direction is None:
                        raise ValueError("Steering direction missing")
                    source = direction["same"] if cell.condition == "steer_same" else direction["delta"]
                    alpha = {"steer_minus": -cell.alpha, "steer_zero": 0.0}.get(cell.condition, cell.alpha)
                    intervention = dict(
                        site=cell.site,
                        source=source,
                        meta=direction["meta"],
                        kind="steer",
                        random=cell.condition == "steer_random",
                        alpha=alpha,
                    )
                    source_id = direction["source_id"]
                sample = (
                    cell.split == "development"
                    and cell.index == 0
                    and intervention is not None
                    and intervention_calls == 0
                )
                grasp_map = (
                    cell.family[0] == "G"
                    and cell.condition == "baseline"
                    and cell.split == "development"
                    and cell.index == 0
                    and phase_calls == 1
                    and scene.eligible
                )
                capture = (cell.site,) if sample else ()
                if grasp_map:
                    capture = tuple(cfg.sites)
                output = self.policy.forward(
                    obs, noise, capture=capture, intervention=intervention, attention=grasp_map
                )
                forwards += 1
                inference_seconds += output["seconds"]
                trace = output["trace"]
                intervention_calls += trace.patches
                if trace.patches:
                    first_intervention = (
                        scene.step_number if first_intervention is None else first_intervention
                    )
                    last_intervention = scene.step_number
                if sample or grasp_map:
                    sample_folder = self.out / "diagnostics" / cell.family
                    sample_folder /= f"rollout_samples/{cell.key}/attempt_{attempt}"
                if sample:
                    save_trace(sample_folder / f"sample_{chunk:03}_recipient", trace)
                    if source_trace:
                        save_trace(sample_folder / f"sample_{chunk:03}_source", source_trace)
                if grasp_map:
                    save_trace(sample_folder / "postgrasp_attention", trace)
                if cfg.save_images:
                    self.save_images(folder, obs, f"{chunk:03}_recipient")
                action = output["actions"]
                if cfg.execute > len(action):
                    raise ValueError("execute exceeds the checkpoint valid action horizon")
                record = dict(
                    chunk=chunk,
                    step=scene.step_number,
                    noise_seed=noise,
                    instruction=next(iter(obs["language"].values()))[0][0],
                    input_bits=recipient,
                    source_id=source_id,
                    source_forward=shadow_record,
                    source_factor=cell.family[0] if intervention else None,
                    tensor_shape=list(intervention["source"].shape) if intervention else None,
                    token_mask=(
                        "last_40_action_tokens"
                        if cell.site.startswith("dit_")
                        else "all_valid_context_tokens"
                    )
                    if intervention
                    else None,
                    donor_bits=donor,
                    source_paraphrase=paraphrase,
                    intervention=bool(intervention),
                    phase_active=active,
                    site=cell.site,
                    denoise_step=cfg.denoise_step,
                    alpha=intervention.get("alpha", 1.0) if intervention else None,
                    rho=trace.rho,
                    patches=trace.patches,
                    actions=action.tolist(),
                    decoded=output["decoded"].tolist(),
                    normalized=output["normalized"][0, : len(action), :7].tolist(),
                    seconds=output["seconds"],
                    peak_cuda_gib=output["peak_cuda_gib"],
                    **scene.snapshot(),
                )
                append_json(folder / "calls.jsonl", record)
                log_step = (attempt - 1) * (cfg.max_steps + 1) + scene.step_number
                mlflow.log_metrics(
                    dict(
                        call_seconds=output["seconds"],
                        peak_cuda_gib=output["peak_cuda_gib"],
                        intervention_calls=intervention_calls,
                        control_step=scene.step_number,
                    ),
                    step=log_step,
                )
                if trace.rho is not None:
                    mlflow.log_metric("rho", trace.rho, step=log_step)
                LOGGER.info(
                    "call %s step=%d edit=%d seconds=%.3f",
                    cell.key,
                    scene.step_number,
                    trace.patches,
                    output["seconds"],
                )
                for a in action[: cfg.execute]:
                    previous_eef = np.asarray(scene.obs["robot0_eef_pos"]).copy()
                    scene.step(a)
                    displacement = float(
                        np.linalg.norm(np.asarray(scene.obs["robot0_eef_pos"]) - previous_eef)
                    )
                    eef_path_m += displacement
                    eef_max_step_m = max(eef_max_step_m, displacement)
                    intervention_steps += int(trace.patches > 0)
                    if scene.first_grasp is not None and grasp_eef is None:
                        grasp_eef = np.asarray(scene.snapshot()["eef"])
                    outcome = scene.outcome(*recipient)
                    append_json(
                        folder / "steps.jsonl",
                        dict(step=scene.step_number, action=a.tolist(), outcome=outcome, **scene.snapshot()),
                    )
                    if writer:
                        writer.append_data(scene.frame())
                    if outcome["select0"] or outcome["select1"] or scene.step_number >= cfg.max_steps:
                        break
                    # Replan at the phase boundary in every condition, including baseline.
                    # Do not execute the rest of a pre-grasp chunk after the grasp transition.
                    if scene.eligible != active:
                        break
                chunk += 1
                if outcome["select0"] or outcome["select1"]:
                    break
            result = dict(
                **dataclasses.asdict(cell),
                scene_id=scene_id,
                key=cell.key,
                **outcome,
                steps=scene.step_number,
                max_steps=cfg.max_steps,
                chunks=chunk,
                intervention_calls=intervention_calls,
                intervention_steps=intervention_steps,
                intervention_start_step=first_intervention,
                intervention_last_call_step=last_intervention,
                eef_path_m=eef_path_m,
                eef_max_step_m=eef_max_step_m,
                model_forwards=forwards,
                inference_seconds=inference_seconds,
                wall_seconds=time.monotonic() - started,
                peak_cuda_gib=output["peak_cuda_gib"],
                stop_reason=(
                    "base_completed"
                    if outcome["select0"]
                    else "alternative_completed"
                    if outcome["select1"]
                    else "horizon"
                ),
                video=str(video_path) if writer else None,
            )
            if direction:
                eef = grasp_eef if grasp_eef is not None else np.asarray(scene.snapshot()["eef"])
                current = scene.f["targets"][1 ^ scene.parity]
                result["distance_probe"] = "first_grasp" if grasp_eef is not None else "final_no_grasp"
                result["d_current"] = float(np.linalg.norm(eef - np.asarray(initial["poses"][current][:3])))
                result["d_sources"] = [
                    float(np.linalg.norm(eef - np.asarray(s["position"]))) for s in direction["sources"]
                ]
            if writer:
                writer.close()
                writer = None
            atomic_json(folder / "outcome.json", result)
            mlflow.log_artifacts(str(folder), "trace")
            LOGGER.info(
                "episode done %s: success=%s select1=%s steps=%d",
                cell.key,
                result["success"],
                result["select1"],
                result["steps"],
            )
            return result
        except BaseException as exc:
            atomic_json(
                folder / "error.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "scene_seed": scene_id,
                    "partial_steps": scene.step_number if hasattr(scene, "step_number") else 0,
                },
            )
            mlflow.log_artifact(str(folder / "error.json"))
            LOGGER.exception("episode incomplete %s", cell.key)
            raise
        finally:
            if writer:
                writer.close()

    @staticmethod
    def save_images(folder, obs, prefix):
        """Optionally save every model input; sampled diagnostics are independent of this flag."""
        from PIL import Image

        path = folder / "images"
        path.mkdir(exist_ok=True)
        for name, video in obs["video"].items():
            Image.fromarray(video[0, 0]).save(path / f"{prefix}_{name}.jpg", quality=85)

    def cells(self, cells, direction=None):
        """Keep conditions of the same scene adjacent for interpretable partial reports."""
        ordered = sorted(cells, key=lambda c: (seed_for(c.family, c.split, c.index), seed_for(c.key)))
        return [self.episode(c, direction) for c in ordered]

    def close(self):
        """Release the native MuJoCo environment and render context."""
        self.scene.close()
