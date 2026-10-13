FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake linux-libc-dev libegl1 libgl1 libgles2 \
    libgl1-mesa-dri libglib2.0-0 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv CC=gcc CXX=g++ \
    PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    HF_HOME=/hf NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true \
    NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY run.py ./
COPY conf ./conf
COPY src ./src
RUN uv run --no-sync python run.py setup
# No token, credentials or checkpoint weights enter build layers.
ENTRYPOINT ["uv", "run", "--no-sync", "python", "run.py"]
CMD ["smoke"]
