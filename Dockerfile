FROM python:3.13-slim

WORKDIR /app

# Install build dependencies for native packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-pip.txt .
RUN pip install --no-cache-dir -r requirements-pip.txt

COPY . .

# Models are downloaded from HuggingFace at runtime.
# To persist them across container restarts, mount a volume:
#   docker run -v hf_cache:/root/.cache/huggingface ...
VOLUME /root/.cache/huggingface

CMD ["python", "main.py"]
