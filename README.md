# AI Procurement Assistant

An AI-powered conversational assistant that answers natural language questions about procurement data. It interprets user queries, generates MongoDB queries and aggregation pipelines, executes them against real data, and returns clear, formatted answers.

Built with LangChain, Google Gemini, MongoDB, and Streamlit.

## Features

- **Natural language to database queries** -- translates conversational questions into MongoDB `find()`, `aggregate()`, and `distinct()` operations automatically
- **Conversational interface** -- Streamlit chat UI with message history and example query sidebar
- **Flexible query handling** -- supports counting, filtering by date ranges, grouping, sorting, top-N analysis, spending breakdowns, and multi-step analytical questions
- **Agentic tool use** -- the LLM decides which database tool to call, inspects results, and can chain multiple tool calls before formulating a final answer
- **Three-level test suite** -- unit tests (no dependencies), integration tests (mock LLM + real DB), and end-to-end evaluation tests (real LLM + real DB via LangSmith)
- **LangSmith observability** -- full trace trees of every agent run, with LLM inputs/outputs, tool call arguments/results, token counts, and latency metrics

## Example Queries

- "How many orders were created in 2013?"
- "Which quarter had the highest total spending?"
- "What are the top 10 most frequently ordered items?"
- "Total spending by department for fiscal year 2014-2015"
- "Which suppliers have the most orders?"
- "What is the average order value by acquisition type?"

## Architecture

```
app/
├── config.py              # Environment configuration (dotenv)
├── db.py                  # MongoDB connection, find/aggregate/distinct helpers
├── agent.py               # LangChain agent with Gemini LLM and tool definitions
└── streamlit_app.py       # Chat UI

scripts/
└── load_data.py           # CSV → MongoDB data loader with type conversions and indexes

tests/
├── conftest.py            # Shared fixtures (MongoDB connection, mock LLM factory)
├── test_unit_parse_dates.py    # Unit tests for date parsing
├── test_unit_serialize.py      # Unit tests for BSON serialization
├── test_integration_agent.py   # Mocked LLM + real MongoDB
└── test_eval_agent.py          # Real LLM + real MongoDB via LangSmith evaluate()
```

## Setup

### Prerequisites

- Python 3.11+
- Docker (for MongoDB)
- A Google Gemini API key
- (Optional) A LangSmith API key for observability

### 1. Start MongoDB

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[test]"          # for running tests
pip install -e ".[observability]" # for LangSmith
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Load data

Download the [California procurement dataset](https://www.kaggle.com/datasets/sohier/large-purchases-by-the-state-of-ca) and place the CSV in `data/lpca/`, then run:

```bash
python scripts/load_data.py
```

### 5. Run the assistant

```bash
streamlit run app/streamlit_app.py
```

## Running Tests

```bash
pytest tests/test_unit_*.py          # Unit tests only (no external deps)
pytest tests/test_integration_*.py   # Integration tests (needs MongoDB)
pytest -m "not eval"                 # Unit + integration (default CI mode)
pytest -m eval                       # Evaluation tests (needs MongoDB + Gemini API key + LangSmith)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini (via LangChain) |
| Agent framework | LangChain with tool-calling |
| Database | MongoDB |
| UI | Streamlit |
| Observability | LangSmith |
| Testing | pytest |
