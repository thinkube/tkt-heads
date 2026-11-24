# tkt-heads

A server for trained parts on a base model, behind the LLM Gateway: taggers
and choice heads, each a few adapters and heads that run on a pretrained base.
It is the `heads` optional component of Thinkube.

## What it does

The container runs `server.py` (FastAPI) on port 8000, one pod per node. The
pod starts with no model; the gateway loads and unloads parts through its
`/admin` routes, and each base is held once, whatever the number of its parts:
three taggers on XLM-RoBERTa large take one copy of the encoder and their own
adapters, not three encoders.

| Route | What it does |
|---|---|
| `POST /admin/models` `{"model_id"}` | Loads a part: the latest MLflow version of the model, and its base from the base's own MLflow mirror when no part on it is loaded yet |
| `POST /admin/models/unload` `{"model_id"}` | Unloads a part; the last part on a base drops the base |
| `GET /v1/models` | The parts loaded |
| `POST /v1/heads` | A part's answer: the body names the model, and the rest of it is the part's request |
| `GET /health` | |

A part is a model of the catalogue's task `heads`: its folder has a
`meepserve.json` naming its kind (such as `tagger`), its format and its base.
The code that runs it is Meep²'s serving package, `meepserve`; see
`design/serving-package.md` in the meepsquared repository. Requests are
answered one at a time.

## How it reaches a user

It is installed and removed from the Optional Components page in
thinkube-control, with no pods until a part is loaded. `spec.llm_backend` in
`thinkube.yaml` declares it a backend of type `heads` with the driver
`shared-pod`: one pod per node, models loaded into it. A part is registered
with Models' `POST /api/v1/models/parts`, and called through the gateway at
`POST https://llm.<domain>/v1/heads`.

## Image

Built on Thinkube's `ai-inference-base` image (CUDA 13.0, PyTorch 2.9, amd64
and arm64), with Transformers, FastAPI and meepserve from `requirements.txt`.

## Licence

This template is MIT licensed. Each model has its own licence, stated in its
catalogue entry.
