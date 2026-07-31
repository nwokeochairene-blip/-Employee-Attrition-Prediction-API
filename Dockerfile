FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

EXPOSE 8000

# Run the app directly (since you have the if __name__ == "__main__" block)
CMD ["python", "app.py"]