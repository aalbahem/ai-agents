"""Unit tests for agent construction and fallback chain."""

from unittest.mock import patch

from langchain_core.runnables import RunnableWithFallbacks

from app.agent import build_agent


@patch("app.agent.GOOGLE_API_KEY", "fake-key")
@patch("app.agent.GEMINI_MODEL", "gemini-primary")
@patch("app.agent.GEMINI_FALLBACK_MODELS", ["gemini-fallback-1", "gemini-fallback-2"])
def test_build_agent_with_fallbacks():
    agent = build_agent()
    # bind_tools on a RunnableWithFallbacks returns a RunnableWithFallbacks
    # whose .runnable is the primary's RunnableBinding and .fallbacks has one
    # RunnableBinding per fallback model.
    assert isinstance(agent, RunnableWithFallbacks)
    assert len(agent.fallbacks) == 2


@patch("app.agent.GOOGLE_API_KEY", "fake-key")
@patch("app.agent.GEMINI_MODEL", "gemini-primary")
@patch("app.agent.GEMINI_FALLBACK_MODELS", [])
def test_build_agent_without_fallbacks():
    agent = build_agent()
    assert not isinstance(agent, RunnableWithFallbacks)


@patch("app.agent.GOOGLE_API_KEY", "fake-key")
@patch("app.agent.GEMINI_MODEL", "gemini-primary")
@patch("app.agent.GEMINI_FALLBACK_MODELS", ["gemini-primary"])
def test_build_agent_skips_primary_in_fallbacks():
    """Fallback list that only contains the primary model should produce no fallbacks."""
    agent = build_agent()
    assert not isinstance(agent, RunnableWithFallbacks)
