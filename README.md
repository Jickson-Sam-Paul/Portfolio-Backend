# AI Portfolio Backend

FastAPI backend for an AI-powered portfolio website with streaming responses and a lightweight RAG pipeline designed for Railway deployment constraints.

## Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- Groq API (LLM)
- In-memory TF-IDF retriever (stdlib only)

## Why this RAG design

This project avoids heavy embedding/vector DB dependencies so it deploys reliably on constrained platforms like Railway:

- No local vector database persistence
- No embedding model downloads at startup
- Small dependency set and fast cold starts
- Retrieval index built in memory from `app/data/profile.txt`

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variable:

```bash
export GROQ_API_KEY="your-api-key-here"
```

## Run locally

```bash
uvicorn app.main:app --reload
```

## Deploy on Railway

This repo already includes:

- `railway.toml`
- `Procfile`

### Steps

1. Push to GitHub.
2. Create Railway project from this repository.
3. Set environment variable in Railway:
   - `GROQ_API_KEY`
4. Deploy.

The service starts with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Health check endpoint:

- `GET /health`

## API

### `POST /api/chat`

Request:

```json
{
  "messages": [{ "role": "user", "content": "Tell me about your experience" }]
}
```

Response:

- Streaming `text/plain`

## Retrieval behavior

- Profile source: `app/data/profile.txt`
- The retriever chunks profile text and computes an in-memory sparse index.
- For each query, top relevant chunks are selected and passed to the LLM.
- If evidence is insufficient, the assistant returns:
  - `I don't have that information yet.`

## Project Structure

```text
app/
  api/
    chat.py
    llm.py
    prompt_builder.py
  utils/
    rag_retriever.py
  data/
    profile.txt
  main.py
```
