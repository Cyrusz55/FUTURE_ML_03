FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8000

WORKDIR /app

RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0

COPY requirements.txt /tmp/requirements.txt
RUN grep -v '^torch==' /tmp/requirements.txt > /tmp/requirements-no-torch.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt && \
    python -c "import torch; assert torch.version.cuda is None"

COPY apps/ /app/apps/
COPY machine_learning/ /app/machine_learning/
COPY frontend/ /app/frontend/
COPY model/ /app/model/
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/model /home/app/.cache/huggingface && \
    python - <<'PY'
import json
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from apps import model_loader
from machine_learning import machine_learning

bundle_path = hf_hub_download(
    repo_id=model_loader.HF_REPO_ID,
    filename=model_loader.HF_FILENAME,
    revision=model_loader.HF_REVISION,
    local_dir="/app/model",
)
with np.load(bundle_path, allow_pickle=False) as raw:
    config = json.loads(str(raw["config_json"]))
revision = config.get("embedding_model_revision")
if not isinstance(revision, str) or not revision:
    raise RuntimeError("Embedding model revision is missing")
snapshot_download(
    repo_id=config["embedding_model_id"],
    revision=revision,
)
bundle = machine_learning.load_bundle(machine_learning.BUNDLE_PATH)
embedder = machine_learning.get_embedder(bundle)
embedder(["Made up screening smoke sentence"])
PY

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /home/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import json,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/api/v1/health', timeout=5); d=json.load(r); assert d.get('model_loaded') is True"

CMD ["sh", "-c", "python -m uvicorn apps.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
