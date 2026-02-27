FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
EXPOSE 8004
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8004"]