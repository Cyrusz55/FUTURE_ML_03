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
COPY pyproject.toml /app/pyproject.toml

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/model /home/app/.cache/huggingface && \
    chown -R appuser:appuser /app /home/app

USER appuser

RUN python - <<'PY'
from apps.model_loader import get_bundle
from machine_learning import machine_learning

bundle = get_bundle()
embedder = machine_learning.get_embedder(bundle)
embedder(["Made up screening smoke sentence"])
PY

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import json,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/api/v1/health', timeout=5); d=json.load(r); assert d.get('model_loaded') is True"

CMD ["sh", "-c", "python -m uvicorn apps.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
