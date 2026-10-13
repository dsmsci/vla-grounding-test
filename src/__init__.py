"""Research modules; tracking stays local with MLflow telemetry disabled."""

import os

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
