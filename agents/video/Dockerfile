FROM python:3.10-slim

# FIX: 'libgl1-mesa-glx' no longer exists. Using 'libgl1' and 'libglib2.0-0'
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8003"]
