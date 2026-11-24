# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

# Thinkube's AI inference base: CUDA 13.0 and PyTorch 2.9, amd64 and arm64.
ARG CONTAINER_REGISTRY
FROM ${CONTAINER_REGISTRY}/library/ai-inference-base:cuda13.0-torch2.9-py3.12

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY server.py .

EXPOSE 8000
CMD ["python3", "server.py"]
