FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CMRC_DATA_ROOT=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY assets ./assets
COPY tests ./tests
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "coupled_monotone_risk.cli.audit", "--skip-data"]
