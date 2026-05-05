FROM python:3.12-slim

# libexpat1 is required by rasterio's bundled GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
CMD ["sh", "-c", "hypercorn main:app --bind 0.0.0.0:$PORT"]
