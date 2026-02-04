"""Level 1: Unit tests for app/search.py — embedding, vector, and fuzzy search."""

from unittest.mock import MagicMock, patch

from clients.typesense import embed_text, search_fuzzy, search_similar_values


class TestEmbedText:
    @patch("clients.typesense._get_embedding_model")
    def test_returns_list_of_floats(self, mock_get_model):
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = fake_model

        result = embed_text("hello")

        assert result == [0.1, 0.2, 0.3]
        fake_model.encode.assert_called_once_with("hello")

    @patch("clients.typesense._get_embedding_model")
    def test_converts_numpy_to_plain_list(self, mock_get_model):
        import numpy as np

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_get_model.return_value = fake_model

        result = embed_text("test")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)


def _mock_multi_search(hits):
    """Create a mock Typesense client whose multi_search.perform returns *hits*."""
    mock_client = MagicMock()
    mock_client.multi_search.perform.return_value = {
        "results": [{"hits": hits}]
    }
    return mock_client


class TestSearchSimilarValues:
    @patch("clients.typesense.embed_text")
    @patch("clients.typesense.get_typesense_client")
    def test_returns_matches_with_value_and_score(
        self, mock_client_fn, mock_embed
    ):
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_client_fn.return_value = _mock_multi_search([
            {
                "document": {
                    "field_name": "Department Name",
                    "value": "Health Care Services, Department of",
                },
                "vector_distance": 0.12,
            },
            {
                "document": {
                    "field_name": "Department Name",
                    "value": "Public Health, Department of",
                },
                "vector_distance": 0.25,
            },
        ])

        results = search_similar_values("Health Department", "Department Name", limit=2)

        assert len(results) == 2
        assert results[0]["value"] == "Health Care Services, Department of"
        assert results[0]["score"] == 0.12
        assert results[1]["value"] == "Public Health, Department of"

    @patch("clients.typesense.embed_text")
    @patch("clients.typesense.get_typesense_client")
    def test_filters_by_field_name(self, mock_client_fn, mock_embed):
        mock_embed.return_value = [0.1]
        mock_client_fn.return_value = _mock_multi_search([])

        search_similar_values("Dell", "Supplier Name", limit=3)

        call_args = mock_client_fn.return_value.multi_search.perform.call_args[0][0]
        assert call_args["searches"][0]["filter_by"] == "field_name:=Supplier Name"

    @patch("clients.typesense.embed_text")
    @patch("clients.typesense.get_typesense_client")
    def test_empty_results(self, mock_client_fn, mock_embed):
        mock_embed.return_value = [0.1]
        mock_client_fn.return_value = _mock_multi_search([])

        results = search_similar_values("nonexistent", "Department Name")

        assert results == []


def _mock_text_search(hits):
    """Create a mock Typesense client whose collection search returns *hits*."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.documents.search.return_value = {"hits": hits}
    mock_client.collections.__getitem__.return_value = mock_collection
    return mock_client


class TestSearchFuzzy:
    @patch("clients.typesense.get_typesense_client")
    def test_returns_matches_with_value_and_score(self, mock_client_fn):
        mock_client_fn.return_value = _mock_text_search([
            {
                "document": {
                    "field_name": "Supplier Name",
                    "value": "Pitney Bowes",
                },
                "text_match_info": {"score": 1234},
            },
            {
                "document": {
                    "field_name": "Supplier Name",
                    "value": "Pitney Bowes Inc.",
                },
                "text_match_info": {"score": 1100},
            },
        ])

        results = search_fuzzy("Pitney", "Supplier Name", limit=2)

        assert len(results) == 2
        assert results[0]["value"] == "Pitney Bowes"
        assert results[0]["score"] == 1234

    @patch("clients.typesense.get_typesense_client")
    def test_passes_typo_tolerance_params(self, mock_client_fn):
        mock_client_fn.return_value = _mock_text_search([])

        search_fuzzy("Delat Denta", "Supplier Name")

        search_call = (
            mock_client_fn.return_value
            .collections.__getitem__.return_value
            .documents.search
        )
        search_args = search_call.call_args[0][0]
        assert search_args["num_typos"] == 2
        assert search_args["prefix"] == "true"
        assert search_args["query_by"] == "value"

    @patch("clients.typesense.get_typesense_client")
    def test_filters_by_field_name(self, mock_client_fn):
        mock_client_fn.return_value = _mock_text_search([])

        search_fuzzy("95841", "Supplier Zip Code")

        search_call = (
            mock_client_fn.return_value
            .collections.__getitem__.return_value
            .documents.search
        )
        search_args = search_call.call_args[0][0]
        assert search_args["filter_by"] == "field_name:=Supplier Zip Code"

    @patch("clients.typesense.get_typesense_client")
    def test_empty_results(self, mock_client_fn):
        mock_client_fn.return_value = _mock_text_search([])

        results = search_fuzzy("nonexistent", "Supplier Name")

        assert results == []
