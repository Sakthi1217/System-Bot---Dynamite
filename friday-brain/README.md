# FRIDAY AI Brain 🧠

A headless AI assistant backend powered by **FastAPI**, **CrewAI**, and **LangChain**.

This is the "Brain" of the FRIDAY system - it receives text commands, orchestrates AI agents to process them, and returns structured responses.

## 🎯 Architecture

```
┌─────────────────┐
│  FastAPI Server │ ← Receives HTTP requests
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│   Main (main.py)     │ ← Routes and validates requests
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│   CrewAI Crew        │ ← Orchestrates agents
└────────┬─────────────┘
         │
    ┌────┴──────────┐
    ▼              ▼
┌─────────┐  ┌──────────────┐
│ Research│  │  Synthesizer │
│ Agent   │  │ Agent        │
└────┬────┘  └────┬─────┘
     │            │
     └────┬───────┘
          ▼
      ┌────────┐
      │ Tools  │ ← DuckDuckGo Search
      └────────┘
```

## 📋 Files Overview

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application with `/ask-friday` endpoint |
| `agents.py` | Research & Synthesizer agents with LLM configuration |
| `tasks.py` | Task definitions that agents execute |
| `tools.py` | Tool initialization (DuckDuckGo search) |
| `requirements.txt` | Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add:

**For OpenAI (Default):**
```
OPENAI_API_KEY=sk-your-api-key-here
```

**For Ollama (Local):**
```
OLLAMA_BASE_URL=http://localhost:11434
# First, pull a model: ollama pull neural-chat
```

### 3. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API

**Interactive Docs:**
```
http://localhost:8000/docs
```

**Using curl:**
```bash
curl -X POST http://localhost:8000/ask-friday \
  -H "Content-Type: application/json" \
  -d '{"command": "What are the latest developments in AI?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask-friday",
    json={"command": "What are the latest developments in AI?"}
)
print(response.json())
```

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns the API health status.

### Ask FRIDAY (Main)
```
POST /ask-friday
```

**Request Body:**
```json
{
  "command": "What are the latest developments in AI?"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "The latest developments in AI include..."
}
```

### Get Info
```
GET /info
```
Returns API metadata and available endpoints.

## 🧠 Agent System

### Research Agent
- **Role:** Gathers factual information
- **Capability:** Uses the DuckDuckGo search tool
- **Output:** Comprehensive research findings

### Synthesizer Agent
- **Role:** Formats research into conversational FRIDAY responses
- **Capability:** Synthesizes information in a friendly, conversational tone
- **Output:** Final, engaging response as FRIDAY

## 🔄 Async Processing

The API uses `asyncio` to handle requests without blocking:

```python
# Agents run in a thread pool
result = await loop.run_in_executor(None, crew.kickoff)
```

This ensures the FastAPI server can handle multiple concurrent requests while agents are processing.

## 🧩 LLM Provider Configuration

### OpenAI (Default)
Edit `agents.py`:
```python
# Uncomment for OpenAI
llm_provider="openai",
model="gpt-4",  # or "gpt-3.5-turbo"
```

Requires: `OPENAI_API_KEY` in `.env`

### Ollama (Local LLM)
Edit `agents.py`:
```python
# Uncomment for Ollama
llm_provider="ollama",
model="neural-chat",  # or "mistral", "dolphin-phi", etc.
```

Setup:
```bash
# Install Ollama from https://ollama.ai
ollama pull neural-chat
ollama serve  # Run in another terminal
```

### Anthropic Claude
Edit `agents.py`:
```python
# Uncomment for Anthropic
llm_provider="anthropic",
model="claude-3-opus",
```

Requires: `ANTHROPIC_API_KEY` in `.env`

## 🛠️ Adding New Tools

1. Import from `langchain_community.tools` or create custom tool
2. Add to `TOOLS` dictionary in `tools.py`:

```python
from langchain_community.tools import YourTool

your_tool = YourTool()
TOOLS = {
    "search": search_tool,
    "your_tool": your_tool,  # ← Add here
}
```

3. Reference in tasks/agents as needed

## 📊 Logging

All API activity is logged to console. Adjust log level in code:

```python
logging.basicConfig(level=logging.INFO)  # INFO, DEBUG, WARNING, ERROR
```

## ⚙️ Production Deployment

For production, use a production ASGI server:

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 📦 Example Integration with Rust Client

When integrated with the Rust client (`friday-os-client`):

```rust
// Rust client sends command to FRIDAY Brain
let response = client.post("http://localhost:8000/process-command")
    .json(&json!({"text": "What time is it?"}))
    .send()
    .await?;
```

## 🐛 Troubleshooting

### "OpenAI API key not found"
- Create `.env` file with `OPENAI_API_KEY=sk-...`

### "Connection refused" for Ollama
- Ensure Ollama is running: `ollama serve` in another terminal
- Check Ollama is running on port 11434

### "Module not found"
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check you're in the correct virtual environment

### Slow responses
- Try with a smaller/faster model
- Check internet connection (if using external search)
- Consider caching frequent queries

## 📚 Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [CrewAI Docs](https://crewai.io/)
- [LangChain Docs](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [OpenAI API](https://platform.openai.com/)

## 📝 Notes

- **Async-safe:** Uses `run_in_executor` to prevent blocking the event loop
- **Modular:** Easy to swap LLM providers or add new tools
- **Validated:** Pydantic models ensure input/output correctness
- **Documented:** Comprehensive comments for easy maintenance
