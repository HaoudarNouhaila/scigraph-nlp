# CPU image for the API and the Streamlit UI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui,shacl]" \
    && python -m spacy download en_core_web_sm

COPY scripts ./scripts
COPY configs ./configs
COPY data/samples ./data/samples
COPY reports ./reports

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "scigraph.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
