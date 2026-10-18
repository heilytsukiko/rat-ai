FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ai/pyproject.toml ai/README.md /opt/ai/
COPY ai/rat_ai /opt/ai/rat_ai
RUN pip install --no-cache-dir /opt/ai --extra-index-url https://download.pytorch.org/whl/cpu
COPY ai/tools/infer_detector.py /opt/ai/tools/infer_detector.py
COPY backend/app ./app
RUN useradd --create-home service && mkdir /app/storage && chown service:service /app/storage
ENV ANALYSIS_MODE=experimental
ENV ANALYZER_COMMAND='["python", "/opt/ai/tools/infer_detector.py"]'
USER service
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
