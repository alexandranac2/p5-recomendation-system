# How to Start the API Server

## Quick Start

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Start the Server
```bash
python run_api.py
```

That's it! The server will:
- Initialize the recommendation system (loads vectorstore)
- Start on `http://localhost:8000`
- Auto-reload on code changes (development mode)

## Access the API

Once started, you can access:

- **API Documentation (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Root endpoint**: http://localhost:8000/
- **List all routes**: http://localhost:8000/api/routes
- **Health check**: http://localhost:8000/api/health/

## Production Mode

For production (no auto-reload):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or modify `run_api.py` to set `reload=False`.

## Requirements

Make sure you have:
- ✅ Virtual environment activated
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Vectorstore exists at `alexs_vectorstore/`
- ✅ `OPENAI_API_KEY` set in environment (for LLM explanations)

## Test the API

After starting, test with:

```bash
# Test with curl
curl -X POST "http://localhost:8000/api/recommendations/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Best running shoes under $200"}'

# Or use the test script
python test_recommendations.py
```

