FROM python:3.10-slim

WORKDIR /app

# ============ 1. Install system dependencies (minimalis) ============
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ============ 2. Copy requirements ============
COPY requirements.txt .

# (Opsional) Gunakan mirror PyPI lebih cepat
# RUN pip config set global.index-url https://pypi.org/simple
# atau: RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ============ 3. Install dependencies dengan timeout & retry ============
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --default-timeout=600 --retries=10 -r requirements.txt

# ============ 4. Copy app code ============
COPY . .

# ============ 5. Create necessary directories ============
RUN mkdir -p data monitoring/prometheus_data

# ============ 6. Expose ports ============
EXPOSE 8501 8000

# ============ 7. Default command ============
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
