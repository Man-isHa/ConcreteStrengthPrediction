FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app.py ./
COPY inference ./inference
COPY models ./models
COPY web ./web

EXPOSE 10000
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-10000}"]