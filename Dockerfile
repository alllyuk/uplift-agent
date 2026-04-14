FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/app

WORKDIR /app

# System deps:
#   graphviz  — runtime binary required by pydot/pyvis for graph rendering
#   curl      — used by HEALTHCHECK
#   libgomp1  — OpenMP runtime needed by faiss / sklearn wheels on slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
        graphviz \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Persistent data dirs — mount host volumes to these paths at runtime.
# Paths match PathsConfig in sme_causal/core/config.py (parents[2] = /app).
RUN mkdir -p \
        /app/artifacts \
        /app/rag_data \
        /app/causal_outputs \
        /app/reports \
    && useradd -m -u 1000 -s /bin/bash app \
    && chown -R app:app /app

USER app

VOLUME ["/app/artifacts", \
        "/app/rag_data", \
        "/app/causal_outputs", \
        "/app/reports"]

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "sme_causal/app/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
