"""Shared test utilities."""


def make_mock_llm(responses):
    """Create a mock LLM that returns predetermined responses in sequence.

    Args:
        responses: List of AIMessage objects to return on successive invoke() calls.

    Returns:
        A mock object with an invoke() method.
    """

    class _MockLLM:
        def __init__(self, responses):
            self._responses = list(responses)
            self._call_count = 0

        def invoke(self, messages):
            idx = min(self._call_count, len(self._responses) - 1)
            self._call_count += 1
            return self._responses[idx]

        @property
        def call_count(self):
            return self._call_count

    return _MockLLM(responses)
