# Junction 2026 starter

```
Frontend (React + Vite)  ->  Backend (FastAPI)  ->  Model service (swappable)
                                    |
                                    +-> Database (SQLite, or Postgres)
```

The model is behind ONE function, `call_model()` in `backend/app/providers.py`.
Change `PROVIDER` in `.env` and nothing else changes.

| PROVIDER    | Use it for                                                    |
|-------------|---------------------------------------------------------------|
| `echo`      | Testing the pipeline with no keys                             |
| `gemini`    | Fast baseline (personal Google AI Studio key)                 |
| `openai`    | Baseline / sponsor credits                                    |
| `anthropic` | Baseline / sponsor credits                                    |
| `local`     | YOUR model: Ollama on the laptop, vLLM on Modal/Verda, Featherless |

## First run (5 minutes)

```bash
cp .env.example .env

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_basics.py                            # offline checks
uvicorn app.main:app --reload --port 8000              # http://localhost:8000/docs

# Frontend (second terminal)
cd frontend
npm install
npm run dev                                            # http://localhost:5173

# Whole chain (third terminal)
python scripts/smoke_test.py                           # echo
python scripts/smoke_test.py http://localhost:8000 gemini
```

## Add a real model

1. **Gemini:** put `GEMINI_API_KEY` in `.env`, set `PROVIDER=gemini`, restart the backend.
2. **Ollama (laptop):** `ollama pull llama3.1:8b`, then `LOCAL_BASE_URL=http://localhost:11434/v1`, `PROVIDER=local`.
3. **Modal (cloud GPU):** `pip install modal && modal setup`, edit `MODEL` in `model_service/modal_vllm.py`,
   `modal deploy model_service/modal_vllm.py`, then set `LOCAL_BASE_URL` to the printed URL + `/v1`.
4. **Fine-tuned model:** train with `finetune/train_lora.py`, push to Hugging Face, put the repo name in `modal_vllm.py`.

## Postgres instead of SQLite

`docker compose up -d db`, then set
`DATABASE_URL=postgresql://junction:junction@localhost:5432/junction` in `.env`.

## Deploy for the demo

- Frontend: push to GitHub, import in Vercel, set `VITE_API_URL` to your backend URL.
- Backend: Docker anywhere (Hostinger, Vultr, Aiven apps), or on the laptop through
  `cloudflared tunnel --url http://localhost:8000` as a backup.

## Theme

All colors and fonts are the variables at the top of `frontend/src/styles.css`.

## Status of this template

Tested offline: request/response builders, echo provider, SQLite storage (`backend/tests`).
NOT tested: FastAPI routes, the React build, Modal and the fine-tune script. Run the first-run steps
above before the event so any version problem shows up now.
