"""Local MLflow is the authoritative store for runs, episode completion and metrics."""

import contextlib
import dataclasses
import fcntl
import hashlib
import logging
import numbers
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from omegaconf import OmegaConf

from .core import ROOT, PINS, digest, project_path

LOGGER = logging.getLogger(__name__)


def configure(cfg):
    """Use a local SQLite backend unless MLFLOW_TRACKING_URI explicitly overrides it."""
    directory = project_path(cfg.tracking.directory)
    directory.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{directory / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.tracking.experiment)
    if experiment is None:
        # MLflow resolves local artifact paths; open the UI in the same runtime as the writer.
        experiment_id = client.create_experiment(
            cfg.tracking.experiment, artifact_location=str(directory / "artifacts")
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return uri, experiment_id


def find_runs(experiment_id, query):
    """Iterate MLflow's paginated run search, including matrices exceeding 1000 episodes."""
    client, token = MlflowClient(), None
    while True:
        page = client.search_runs([experiment_id], query, max_results=1000, page_token=token)
        yield from page
        token = page.token
        if not token:
            break


@contextlib.contextmanager
def experiment_run(cfg, stage, runtime):
    """Resume an MLflow parent run only when code, configuration and runtime still match."""
    out = project_path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    configure(cfg)
    values = OmegaConf.to_container(cfg, resolve=True)
    scientific = {
        k: v
        for k, v in values.items()
        if k not in {"output", "videos", "tracking", "save_videos", "save_images"}
    }
    code = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src").glob("*.py"))
    }
    for name in ("run.py", "pyproject.toml", "uv.lock"):
        code[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    signature = digest(dict(config=scientific, code=code, pins=PINS))
    label = digest(str(cfg.output))
    experiment_id = mlflow.get_experiment_by_name(cfg.tracking.experiment).experiment_id
    matches = list(find_runs(experiment_id, f"tags.kind = 'protocol' and tags.output = '{label}'"))
    if len(matches) > 1:
        raise ValueError("Multiple parent runs for this output; select a new output directory.")
    parent = matches[0] if matches else None
    if parent is None and any(
        (out / name).exists() for name in ("episodes", "decisions", "diagnostics", "scenes")
    ):
        raise ValueError(
            "Output has scientific artifacts without its MLflow parent; restore MLflow or use a new output."
        )
    if parent and parent.data.tags.get("signature") != signature:
        raise ValueError("Scientific configuration, code or dependencies changed; use a new output.")
    with (out / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mlflow.start_run(
            run_id=parent.info.run_id if parent else None, run_name=Path(cfg.output).name
        ) as run:
            if parent:
                previous = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/runtime.json")
                for key in ("python", "torch", "cuda_runtime", "gpu", "driver", "packages"):
                    if previous.get(key) != runtime.get(key):
                        raise ValueError(f"Runtime changed ({key}); use a new output.")
            else:
                mlflow.log_params({k: v for k, v in scientific.items() if not isinstance(v, dict)})
                mlflow.log_dict(
                    dict(config=scientific, code=code, pins=PINS, signature=signature), "manifest.json"
                )
                mlflow.log_dict(runtime, "runtime.json")
                mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
                for path in sorted((ROOT / "src").glob("*.py")):
                    mlflow.log_artifact(str(path), "source/src")
                mlflow.log_artifact(str(ROOT / "pyproject.toml"), "source")
                mlflow.log_artifact(str(ROOT / "uv.lock"), "source")
                mlflow.log_artifact(str(ROOT / "run.py"), "source")
            # Portable provenance next to report tables; MLflow remains authoritative.
            client = MlflowClient()
            for name in ("config.yaml", "manifest.json", "runtime.json"):
                client.download_artifacts(run.info.run_id, name, str(out))
            mlflow.set_tags(dict(kind="protocol", output=label, signature=signature, last_stage=stage))
            try:
                yield run.info.run_id
                mlflow.set_tag("last_exit", "stage complete")
            except KeyboardInterrupt:
                mlflow.set_tag("last_exit", "interrupted; repeat the same command")
                raise
            finally:
                for name in ("decisions", "diagnostics", "scenes"):
                    if (out / name).is_dir():
                        mlflow.log_artifacts(str(out / name), name)
                if (out / "execution.log").is_file():
                    mlflow.log_artifact(str(out / "execution.log"))


def completed_results(parent_id=None):
    """Read only completed MLflow episode runs; failed/interrupted attempts are excluded."""
    active = mlflow.active_run()
    parent_id = parent_id or active.info.run_id
    experiment_id = (
        active.info.experiment_id if active else MlflowClient().get_run(parent_id).info.experiment_id
    )
    query = f"tags.mlflow.parentRunId = '{parent_id}' and tags.kind = 'episode' and attributes.status = 'FINISHED'"
    rows = []
    for run in find_runs(experiment_id, query):
        if run.data.tags.get("completed") != "true":
            continue
        row = {**run.data.params, **run.data.metrics, "mlflow_run_id": run.info.run_id}
        for name in ("index", "seed"):
            row[name] = int(row[name])
        row["alpha"] = float(row["alpha"])
        row["key"] = run.data.tags["key"]
        row["stop_reason"] = run.data.tags.get("stop_reason")
        row["T"] = run.data.tags.get("object")
        goal = run.data.tags.get("goal_index")
        row["G"] = int(goal) if goal and goal not in {"none", "None"} else None
        rows.append(row)
    return rows


@contextlib.contextmanager
def episode_run(cell):
    """Create or resume one nested MLflow episode and mark completion only after logging."""
    parent = mlflow.active_run()
    query = f"tags.mlflow.parentRunId = '{parent.info.run_id}' and tags.key = '{cell.key}'"
    matches = list(find_runs(parent.info.experiment_id, query))
    if len(matches) > 1:
        raise ValueError(f"Duplicate MLflow episode: {cell.key}")
    previous = matches[0] if matches else None
    attempt = int(previous.data.tags.get("attempt", 0)) + 1 if previous else 1
    with mlflow.start_run(
        run_id=previous.info.run_id if previous else None, run_name=cell.key, nested=True
    ) as run:
        mlflow.set_tags(dict(kind="episode", key=cell.key, attempt=attempt, completed="false"))
        mlflow.log_params(dataclasses.asdict(cell))
        yield run.info.run_id, attempt


def log_outcome(result):
    """Log scalar outcome/cost metrics and a full outcome artifact in the active episode."""
    metrics = {
        k: float(v)
        for k, v in result.items()
        if isinstance(v, numbers.Real) and k not in {"index", "seed", "alpha"}
    }
    mlflow.log_metrics(metrics)
    mlflow.log_dict(result, "outcome.json")
    mlflow.set_tags(
        dict(
            completed="true",
            stop_reason=result["stop_reason"],
            object=result.get("T") or "none",
            goal_index=str(result.get("G")),
        )
    )
