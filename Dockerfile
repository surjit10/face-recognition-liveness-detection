FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.lock ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.lock \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN useradd --create-home --uid 1000 faceguard \
    && chown -R faceguard:faceguard /app
USER faceguard

ENTRYPOINT ["python", "scripts/run_demo.py"]
CMD ["--config", "configs/runtime.yaml", "--check-only"]
