# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

# Thinkube's vLLM base: vLLM 0.32 with CUDA 13.0 and PyTorch 2.13, amd64 and arm64. The choice kind's vllm
# engine holds a base in vLLM for generation and the choice head's states; its Transformers and FastAPI are
# vLLM's.
ARG CONTAINER_REGISTRY
FROM ${CONTAINER_REGISTRY}/library/vllm-base:0.32.0-cuda13.0-py3.12

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY server.py .

EXPOSE 8000
CMD ["python3", "server.py"]
