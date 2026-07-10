FROM python:3.10-slim

WORKDIR /app

# System libs: cv2 needs libGL + libglib; the preprocessed-video recorder shells
# out to ffmpeg. Without these the container fails at runtime, not build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Serving-only deps (see requirements-serving.txt). Model weights and .env are
# NOT baked (excluded by .dockerignore); they are mounted / injected at runtime.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

EXPOSE 8000

# API role. The worker service overrides this with `python -m api.worker`.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
