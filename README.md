# Text-to-SQL LoRA — Production Project

Fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with LoRA to translate a
table schema + natural-language question into SQL, then serves it behind a
FastAPI endpoint.

Dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
(schema, question, SQL-answer triples).

## Architecture

```
                 ┌────────────────┐
  train.py  ───▶ │  LoRA adapter   │ ──▶ inference.py (merge) ──▶ merged weights
                 │  (artifacts/)   │
                 └────────────────┘
                                                    │
                                                    ▼
                                          api/main.py (FastAPI)
                                                    │
                                            POST /generate
                                          {schema, question} -> {sql}
```

- **src/config.py** — typed config (dataclasses), overridable via `configs/default.yaml`
- **src/data.py** — dataset loading + chat-template prompt formatting
- **src/train.py** — LoRA fine-tuning entrypoint (CLI, optional W&B logging)
- **src/evaluate.py** — exact-match accuracy **and execution accuracy**
  (builds a throwaway SQLite DB from the schema and actually runs both the
  gold and predicted SQL against it — this is the metric that matters for
  text-to-SQL, since two queries can differ textually but return the same
  result)
- **src/inference.py** — merges the LoRA adapter into base weights for
  faster serving, and a `SQLPredictor` class used by the API
- **api/main.py** — FastAPI service (`/health`, `/generate`, and serves the frontend)
- **frontend/** — a small static UI (no build step) that calls `/generate` and shows the SQL result
- **tests/** — fast, no-GPU unit tests (config, SQL normalization, SQLite
  harness) that run in CI on every push
- **.github/workflows/ci.yml** — lint + test + Docker build on every PR

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config configs/default.yaml
# override anything on the CLI, e.g.:
python -m src.train --config configs/default.yaml --epochs 3 --report-to wandb
```

Adapter weights are saved to `artifacts/adapter/` (see `model.adapter_dir`
in the config).

## Evaluate

```bash
python -m src.evaluate --config configs/default.yaml \
    --adapter-dir artifacts/adapter --n 200
```

Prints exact-match accuracy, execution accuracy, and execution error rate,
and writes per-example predictions to `eval_results.json`. Run it once with
`--adapter-dir` omitted to get the **base model** baseline for comparison.

## Merge weights for serving

```bash
python -m src.inference --config configs/default.yaml
```

Writes merged weights to `artifacts/merged/` (faster load, no PEFT
overhead at inference time).

## Serve locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in a browser for the UI — enter a schema and
a question (or click one of the example chips), and it calls `/generate`
and shows the SQL with latency. `Cmd/Ctrl+Enter` also runs it.

Or hit the API directly:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "schema": "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);",
        "question": "List the names of employees in the Engineering department earning more than 100000."
      }'
```

## Run with Docker

```bash
docker compose up --build
```

The container expects merged (or adapter) weights mounted at
`/app/artifacts` — build/merge them locally first, or pull from wherever
you store model artifacts (S3, Hugging Face Hub, etc.) as part of your
deploy step.

## Deploying

- **Hugging Face Spaces** — push the merged model to the HF Hub, put a
  small Gradio/Streamlit front-end in front of this FastAPI backend (or
  use a Space's Docker SDK directly with this Dockerfile).
- **AWS** — push the image to ECR and run it on an EC2/Lightsail instance
  or ECS; store `artifacts/merged/` in S3 and pull it on container start.

## Results

_Fill in after running `evaluate.py` on base vs fine-tuned:_

| Model              | Exact Match | Execution Accuracy |
|--------------------|:-----------:|:-------------------:|
| Base TinyLlama-1.1B|     —       |          —           |
| + LoRA fine-tune   |     —       |          —           |

## License

MIT (adjust as needed).
