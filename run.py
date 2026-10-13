#!/usr/bin/env python3
"""One Fire entry point; Hydra supplies experiment defaults."""

import os
import json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import fire

from src.cli import main

if __name__ == "__main__":
    fire.Fire(main, serialize=lambda result: json.dumps(result, ensure_ascii=False, indent=2))
