FROM python:3.14-trixie

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git nano vim wget \
    && rm -rf /var/lib/apt/lists/*

# copy and install package
COPY . .
RUN python -m pip install --no-cache-dir -e ".[dev]"
