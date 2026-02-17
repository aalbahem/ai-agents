"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai", "gemini", or "ollama"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "procurement")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "purchases")
VIEW_NAME = os.getenv("VIEW_NAME", "purchase_orders")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))

_fallback_raw = os.getenv(
    "GEMINI_FALLBACK_MODELS",
    "gemini-2.5-flash-lite-preview-06-17,gemini-2.0-flash",
)
GEMINI_FALLBACK_MODELS: list[str] = [
    m.strip() for m in _fallback_raw.split(",") if m.strip()
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "localhost")
TYPESENSE_PORT = os.getenv("TYPESENSE_PORT", "8108")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "xyz")
TYPESENSE_COLLECTION = os.getenv("TYPESENSE_COLLECTION", "field_values")
