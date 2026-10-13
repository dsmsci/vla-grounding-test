"""Paired rollout effects using MLflow records and SciPy confidence intervals."""

from itertools import product

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import binomtest, bootstrap

from .core import atomic_json, project_path, read_json
from .tracking import completed_results

METRICS = ("success", "select0", "select1", "T_correct", "G_correct", "R", "preserve", "choice1")
GROUPS = ("family", "split", "contrast", "condition", "site", "alpha")


def paired(rows, terms, metric, seed=17, repeats=10000):
    """Bootstrap complete paired scenes; uncertainty is conditional on the one policy seed."""
    lookups = []
    for _, selector in terms:
        lookup = {}
        for row in rows:
            if all(row[k] == v for k, v in selector.items()):
                key = int(row["scene_id"])
                if key in lookup:
                    raise ValueError(f"Duplicate matched scene {key}: refine selector {selector}")
                lookup[key] = float(row[metric])
        lookups.append(lookup)
    candidates = set().union(*(set(lookup) for lookup in lookups))
    complete = sorted(set.intersection(*(set(lookup) for lookup in lookups)))
    values = [
        sum(coefficient * lookup[s] for (coefficient, _), lookup in zip(terms, lookups)) for s in complete
    ]
    ci = None
    if len(values) >= 2:
        interval = bootstrap(
            (np.asarray(values),),
            np.mean,
            method="percentile",
            n_resamples=repeats,
            rng=np.random.default_rng(seed),
        ).confidence_interval
        ci = [float(interval.low), float(interval.high)]
    return dict(
        metric=metric,
        estimate=float(np.mean(values)) if values else None,
        ci95=ci,
        scenes=len(complete),
        excluded_incomplete_scenes=len(candidates) - len(complete),
        episodes_per_condition=len(complete),
        terms=terms,
    )


def coverage(cfg, rows):
    """Count the full held-out matrix, including conditions not yet scheduled."""
    result = []
    out = project_path(cfg.output)
    for family in cfg.families:
        path = out / "decisions" / family / "frozen.json"
        if not path.exists():
            result.append(dict(family=family, expected=None, completed=0, status="development not frozen"))
            continue
        frozen = read_json(path)
        expected = set()
        for contrast, gate in frozen["patches"].items():
            if gate is None:
                continue
            for i in range(frozen["test_scenes"]):
                expected.update((contrast, c, "", 1.0, i) for c in ("baseline", "alternate"))
                expected.update(
                    (contrast, c, site, 1.0, i)
                    for site, c in product(
                        (gate["site"],), ("patch_to_alt", "patch_to_base", "patch_same", "patch_random")
                    )
                )
        if frozen["mixed"]:
            expected.update(
                ("mixed", prefix + f"{t}{v}", site, 1.0, i)
                for prefix, site in (("mixed_", ""), ("mixedpatch_", "context"))
                for t, v, i in product(range(2), range(2), range(frozen["test_scenes"]))
            )
        steering = frozen["steering"]
        if steering and steering["alpha"] is not None:
            expected.update(
                ("text", c, steering["site"], steering["alpha"], i)
                for c, i in product(
                    ("steer", "steer_minus", "steer_same", "steer_random", "steer_zero"),
                    range(frozen["test_scenes"]),
                )
            )
        actual = {
            (r["contrast"], r["condition"], r["site"], r["alpha"], r["index"])
            for r in rows
            if r["family"] == family and r["split"] == "test"
        }
        result.append(
            dict(
                family=family,
                expected=len(expected),
                completed=len(expected & actual),
                status="skipped: no admitted patch/steering"
                if not expected
                else "complete"
                if expected <= actual
                else "incomplete",
            )
        )
    return result


def report(cfg, parent_id):
    """Write compact tables from completed MLflow runs and upload them to the parent."""
    out = project_path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = completed_results(parent_id)
    rates, effects = [], []
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "episodes.csv", index=False)
        for key, group in frame.groupby(list(GROUPS), dropna=False):
            n, k = len(group), int(group["success"].sum())
            interval = binomtest(k, n).proportion_ci(method="wilson")
            rates.append(
                dict(zip(GROUPS, key))
                | dict(
                    n=n,
                    successes=k,
                    success_ci95=[float(interval.low), float(interval.high)],
                    **{m: float(group[m].mean()) for m in METRICS},
                )
            )
        for (family, split, contrast), group in frame.groupby(["family", "split", "contrast"]):
            subset = group.to_dict("records")
            comparisons = []
            for condition, site, alpha in sorted(set(zip(group["condition"], group["site"], group["alpha"]))):
                if condition.startswith(("patch_", "steer")):
                    target = dict(condition=condition, site=site, alpha=alpha)
                    control = dict(
                        condition="alternate" if condition == "patch_to_base" else "baseline",
                        site="",
                        alpha=1.0,
                    )
                    comparisons.append((f"{condition}:{site}:{alpha}", [(1, target), (-1, control)]))
                    if condition == "patch_to_alt":
                        for c in ("patch_same", "patch_random"):
                            comparisons.append(
                                (
                                    f"patch_to_alt-{c}:{site}",
                                    [(1, target), (-1, dict(condition=c, site=site, alpha=1.0))],
                                )
                            )
            for prefix, site in (("mixed_", ""), ("mixedpatch_", "context")):
                if any(r["condition"].startswith(prefix) for r in subset):
                    comparisons.append(
                        (
                            prefix + "D",
                            [
                                (weight, dict(condition=prefix + bits, site=site, alpha=1.0))
                                for weight, bits in ((1, "11"), (-1, "10"), (-1, "01"), (1, "00"))
                            ],
                        )
                    )
            for name, terms in comparisons:
                for metric in METRICS:
                    value = paired(subset, terms, metric, cfg.seed, cfg.bootstrap)
                    effects.append(
                        dict(family=family, split=split, contrast=contrast, comparison=name, **value)
                    )
    result = dict(
        completed=len(rows),
        seed=cfg.seed,
        rates=rates,
        effects=effects,
        coverage=coverage(cfg, rows),
        interpretation="Pilot; intervals condition on one policy seed.",
    )
    atomic_json(out / "summary.json", result)
    pd.DataFrame(rates).to_csv(out / "rates.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "effects.csv", index=False)
    client = MlflowClient()
    for name in ("episodes.csv", "rates.csv", "effects.csv", "summary.json"):
        if (out / name).is_file():
            client.log_artifact(parent_id, str(out / name), "tables")
    client.log_metric(parent_id, "completed_episodes", len(rows))
    for row in rates:
        key = "/".join(str(row[k]) if row[k] != "" else "none" for k in GROUPS)
        client.log_metric(parent_id, f"SR/{key}", row["success"])
    return {k: v for k, v in result.items() if k not in {"rates", "effects"}}
