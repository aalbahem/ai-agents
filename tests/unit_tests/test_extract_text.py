"""Level 1: Unit tests for _extract_text() from app/agent.py."""

from app.agent import _extract_text


class TestExtractTextPlainString:
    def test_returns_string_unchanged(self):
        assert _extract_text("hello world") == "hello world"

    def test_empty_string(self):
        assert _extract_text("") == ""


class TestExtractTextContentBlocks:
    """LLM responses may come back as a list of typed content blocks."""

    def test_single_text_block(self):
        content = [{"type": "text", "text": "The answer is 42."}]
        assert _extract_text(content) == "The answer is 42."

    def test_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]
        assert _extract_text(content) == "Part one.\nPart two."

    def test_ignores_extras_and_metadata(self):
        """Gemini responses include extras with signatures — these must be stripped."""
        content = [
            {
                "type": "text",
                "text": "There were 6953 orders.",
                "extras": {"signature": "CqYDAXLI2nw..."},
            }
        ]
        assert _extract_text(content) == "There were 6953 orders."

    def test_ignores_non_text_blocks(self):
        content = [
            {"type": "text", "text": "Answer."},
            {"type": "image", "data": "..."},
        ]
        assert _extract_text(content) == "Answer."

    def test_list_of_plain_strings(self):
        content = ["hello", "world"]
        assert _extract_text(content) == "hello\nworld"

    def test_empty_list(self):
        assert _extract_text([]) == ""


class TestExtractTextFallback:
    def test_non_string_non_list(self):
        assert _extract_text(123) == "123"

    def test_none(self):
        assert _extract_text(None) == "None"
