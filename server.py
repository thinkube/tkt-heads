# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: MIT

"""Trained parts on their base models, many in one pod, behind the LLM gateway (backend type heads, driver shared-pod).

A part (a tagger, a choice head) is a model of the catalogue's task heads: a
folder in MLflow whose meepserve.json names its kind and its base. The pod
starts empty; the gateway loads and unloads parts through its /admin routes,
and each base is loaded once, from its own MLflow mirror, when its first part
comes, and dropped when its last part goes.

    POST /admin/models           {"model_id": ...}  load a part
    POST /admin/models/unload    {"model_id": ...}  unload a part
    GET  /v1/models                                the parts loaded
    POST /v1/heads              {"model": ..., ...} a part's answer; the rest
                                                   of the body is its kind's request
    GET  /health

Requests are answered one at a time: the parts of a base switch its adapters.
"""

import gc
import os
import sys
import threading
import time

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException

MLFLOW_URL = "http://mlflow.mlflow.svc.cluster.local:5000"


def required(name: str, why: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"{name} is not set: {why}")
    return value


MLFLOW_ENV = {n: required(n, "the component's MLflow credentials Secret provides it") for n in (
    "MLFLOW_KEYCLOAK_TOKEN_URL", "MLFLOW_KEYCLOAK_CLIENT_ID", "MLFLOW_CLIENT_SECRET",
    "MLFLOW_AUTH_USERNAME", "MLFLOW_AUTH_PASSWORD")}


class Refused(Exception):
    """A part cannot be loaded; the message says why."""


def model_dir(model_id: str) -> str:
    """The folder of the latest MLflow version of the model `model_id`."""
    resp = httpx.post(MLFLOW_ENV["MLFLOW_KEYCLOAK_TOKEN_URL"], data={
        "grant_type": "password", "client_id": MLFLOW_ENV["MLFLOW_KEYCLOAK_CLIENT_ID"],
        "client_secret": MLFLOW_ENV["MLFLOW_CLIENT_SECRET"], "username": MLFLOW_ENV["MLFLOW_AUTH_USERNAME"],
        "password": MLFLOW_ENV["MLFLOW_AUTH_PASSWORD"], "scope": "openid",
    }, timeout=30)
    if resp.status_code != 200:
        raise Refused(f"Keycloak refused the MLflow token: HTTP {resp.status_code}: {resp.text[:200]}")
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    registered = model_id.replace("/", "-")
    resp = httpx.get(f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search",
                     params={"filter": f"name='{registered}'"}, headers=headers, timeout=30)
    resp.raise_for_status()
    versions = resp.json().get("model_versions", [])
    if not versions:
        raise Refused(f"{model_id} is not registered in MLflow as {registered}")
    run_id = max(versions, key=lambda v: int(v["version"]))["run_id"]
    resp = httpx.get(f"{MLFLOW_URL}/api/2.0/mlflow/runs/get", params={"run_id": run_id}, headers=headers, timeout=30)
    resp.raise_for_status()
    experiment_id = resp.json()["run"]["info"]["experiment_id"]
    return f"/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model"


lock = threading.Lock()
parts = {}      # model id -> part
bases = {}      # model id -> base id
engines = {}    # base id -> engine


def load_part(model_id: str) -> dict:
    import meepserve
    if model_id in parts:
        return {"model_id": model_id, "base": bases[model_id], "already": True}
    folder = model_dir(model_id)
    contract = meepserve.read_contract(folder)
    base = contract["base"]
    began = time.time()
    created = base not in engines
    if created:
        engines[base] = meepserve.engine_for(contract, model_dir(base), device="cuda")
    try:
        parts[model_id] = meepserve.load(folder, engine=engines[base])
    except Exception:
        if created and not any(b == base for b in bases.values()):
            drop_engine(base)
        raise
    bases[model_id] = base
    return {"model_id": model_id, "base": base, "base_loaded": created, "seconds": round(time.time() - began, 1)}


def drop_engine(base: str) -> None:
    engine = engines.pop(base)
    if hasattr(engine, "close"):
        engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def unload_part(model_id: str) -> dict:
    if model_id not in parts:
        raise HTTPException(404, f"{model_id} is not loaded here; loaded: {sorted(parts)}")
    parts.pop(model_id).close()
    base = bases.pop(model_id)
    dropped = base not in bases.values()
    if dropped:
        drop_engine(base)
    return {"model_id": model_id, "base": base, "base_dropped": dropped}


if not torch.cuda.is_available():
    sys.exit("No CUDA device is visible to this pod")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": model_id, "object": "model", "owned_by": "thinkube", "task": "heads", "base": bases[model_id]}
        for model_id in sorted(parts)]}


@app.post("/admin/models")
def admin_load(body: dict):
    from meepserve import PartError
    model_id = body.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(422, "model_id names the part to load")
    try:
        with lock:
            return load_part(model_id)
    except (Refused, PartError) as e:
        raise HTTPException(409, str(e))


@app.post("/admin/models/unload")
def admin_unload(body: dict):
    with lock:
        return unload_part(body.get("model_id", ""))


@app.post("/v1/heads")
def heads(body: dict):
    from meepserve import PartError
    name = body.get("model", "")
    with lock:
        part = parts.get(name)
        if part is None:
            raise HTTPException(404, f"{name} is not loaded here; loaded: {sorted(parts)}")
        try:
            return part.answer(body)
        except PartError as e:
            raise HTTPException(400, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
