# The crop & fertiliser recommendation engine, containerised.
#
# Build from THIS directory:
#
#   docker build -t agrosense-recommend .
#
# Deliberately its own image rather than a route bolted onto backend/'s
# service: that service already loads torch (the soil-photo classifier) and
# xgboost, and backend/config.py documents a real SIGSEGV from torch's and
# scikit-learn's OpenMP runtimes sharing one process. This service brings its
# own lightgbm/scikit-learn/scipy stack, so it gets its own process, its own
# image, and its own port (8001 — backend's reading service already owns 8000).

FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

ENV PYTHONUNBUFFERED=1

# Fit the pipeline now, into the image, rather than on the first boot.
#
# The cache key (src/pipeline.py `_cache_key`) hashes the source and the raw
# files' size and mtime, plus the installed library versions — so a cache
# fitted on a laptop never matches in here, and a container that had to fit
# at start-up would spend minutes failing its health check on 1 vCPU. COPY
# keeps file mtimes, so the key computed now is the key the running container
# computes, and it loads this fit at once. Any stale pipeline_*.joblib that
# came in with artifacts/ is deleted by the fit itself.
#
# Then one real recommendation, so an image that cannot answer never ships.
RUN python -c "from src.pipeline import load_pipeline; load_pipeline()" \
    && python -m src.cli --district PUNE --taluka BARAMATI --season Rabi > /dev/null

# Not root. The one place the engine writes is `artifacts/`, and only when its
# cache key no longer matches the code (it then refits and saves the result).
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
    && chown -R app:app /app/artifacts
USER app

EXPOSE 8001

CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
