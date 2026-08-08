FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY data/corpus/ data/corpus/

EXPOSE 8000

CMD ["uvicorn", "rag.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
