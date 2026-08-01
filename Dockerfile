FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Render's PORT environment variable
ENV PORT=10000
EXPOSE $PORT

# Use gunicorn with the correct port
CMD gunicorn --bind 0.0.0.0:$PORT app:app