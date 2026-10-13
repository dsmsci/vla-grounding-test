"""Python Fire commands backed by Hydra composition and standard MLflow tracking."""

import gc
import logging
import subprocess

import mlflow
import numpy as np
import torch
from PIL import Image
from transformers import set_seed

from .core import atomic_json, config as compose_config, digest, project_path
from .runtime import activate, doctor, download, setup
from .tracking import configure, experiment_run, find_runs

LOGGER = logging.getLogger(__name__)
STAGES = ("smoke", "baseline", "diagnose", "patch", "steer", "develop", "test")


def simcheck(cfg):
    """Reset/render every configured family and verify that swapping colors preserves physics."""
    from .scenes import Scene

    out, results = project_path(cfg.output) / "simcheck", {}
    out.mkdir(parents=True, exist_ok=True)
    for family in cfg.families:
        scene = Scene(family, cfg.success_hold_steps)
        try:
            first = scene.reset(cfg.seed)
            obs0 = scene.observation(0, 0)
            state0 = np.array(scene.snapshot()["state"])
            obs1 = scene.observation(0, int(family.endswith("color")))
            state1 = np.array(scene.snapshot()["state"])
            second = scene.reset(cfg.seed)
            reset_error = float(np.max(np.abs(np.array(first["state"]) - second["state"])))
            physics_error = float(np.max(np.abs(state0 - state1)))
            rgb_error = float(np.mean(np.abs(obs0["video"]["image"].astype(float) - obs1["video"]["image"])))
            if reset_error > 1e-10 or physics_error > 1e-10:
                raise RuntimeError(f"Unpaired scene: {family}")
            if family.endswith("color") and rgb_error <= 0:
                raise RuntimeError(f"Color exchange is not visible: {family}")
            for bit, obs in enumerate((obs0, obs1)):
                Image.fromarray(obs["video"]["image"][0, 0]).save(out / f"{family}_v{bit}.png")
            results[family] = dict(
                reset_max_abs=reset_error,
                visual_physics_max_abs=physics_error,
                mean_rgb_change=rgb_error,
                state=first,
            )
        finally:
            scene.close()
    atomic_json(out / "simcheck.json", results)
    return {f: {k: v for k, v in r.items() if k != "state"} for f, r in results.items()}


def plan(cfg):
    """Show episode upper bounds for fixed-site patching and optional steering."""
    rows = {}
    d, t, k = cfg.development, cfg.test, len(cfg.sites)
    for family in cfg.families:
        contrasts = 2 if family.endswith("color") else 1
        rows[family] = dict(
            baseline=2 * d * contrasts,
            development_patch=(2 * k * d + 2 * d + 1) * contrasts,
            development_steering=len(cfg.alphas) * d if family.startswith("T") else 0,
            test_patch=6 * t * contrasts,
            test_steering=5 * t if family.startswith("T") else 0,
            test_mixed=8 * t if family.endswith("color") else 0,
        )
        rows[family]["maximum"] = sum(rows[family].values())
    return dict(
        seed=cfg.seed,
        sites=list(cfg.sites),
        episodes=rows,
        note="Upper bounds if competence/steering gates pass; donor forwards add computation.",
    )


def main(stage="smoke", *overrides, config="mvp"):
    """Run a stage using conf/<config>.yaml and Hydra key=value overrides.

    Examples:
        python run.py smoke
        python run.py develop
        python run.py diagnose --config=color
        python run.py smoke output=results/probe save_videos=false

    Stages: setup, download, doctor, simcheck, plan, smoke, baseline,
            diagnose, patch, steer, develop, test, report, ui.
    """
    cfg = compose_config(config, overrides)
    if stage == "plan":
        return plan(cfg)
    if stage == "ui":
        uri, _ = configure(cfg)
        subprocess.run(
            [
                "mlflow",
                "server",
                "--backend-store-uri",
                uri,
                "--host",
                cfg.tracking.host,
                "--port",
                str(cfg.tracking.port),
                "--workers",
                "1",
            ],
            check=True,
        )
        return
    if stage == "setup":
        setup()
        return {"sources": "ready"}
    if stage == "download":
        download(cfg.families)
        return {"checkpoints": "ready"}
    if stage == "report":
        from .stats import report

        _, experiment_id = configure(cfg)
        label = digest(str(cfg.output))
        matches = list(find_runs(experiment_id, f"tags.kind = 'protocol' and tags.output = '{label}'"))
        if len(matches) != 1:
            raise ValueError("No unique experiment run for this output; run smoke first.")
        return report(cfg, matches[0].info.run_id)
    activate()
    if stage == "doctor":
        return doctor()
    if stage == "simcheck":
        return simcheck(cfg)
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    from .experiment import Experiment
    from .policy import Policy
    from .rollout import Rollout
    from .scenes import FAMILIES
    from .stats import report

    if set(cfg.families) - FAMILIES.keys():
        raise ValueError("Unknown LIBERO family")
    out = project_path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "execution.log")],
        force=True,
    )
    set_seed(cfg.seed, deterministic=True)
    if stage == "test":
        missing = [f for f in cfg.families if not (out / "decisions" / f / "frozen.json").is_file()]
        if missing:
            raise ValueError(f"Finish develop before test for {missing}")
    info = doctor()
    try:
        with experiment_run(cfg, stage, info) as parent_id:
            for family in cfg.families:
                LOGGER.info("model load %s; actual device: %s", family, info)
                policy, engine = None, None
                try:
                    policy = Policy(FAMILIES[family]["suite"], cfg)
                    mlflow.log_metric(f"model_allocated_gib/{family}", torch.cuda.memory_allocated() / 2**30)
                    engine = Rollout(cfg, policy, family)
                    Experiment(engine).run(stage)
                finally:
                    if engine is not None:
                        engine.close()
                    del engine, policy
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            result = report(cfg, parent_id)
            LOGGER.info("stage %s done: %s", stage, result)
            return result
    except KeyboardInterrupt:
        LOGGER.info("Interrupted. Repeat the same command to resume completed MLflow episodes.")
        return {"status": "interrupted"}
