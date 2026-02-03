"""Level 1: Unit tests for scripts/index_typesense.py."""

from scripts.index_typesense import _stable_id


class TestStableId:
    def test_deterministic(self):
        id1 = _stable_id("Department Name", "Health Care Services, Department of")
        id2 = _stable_id("Department Name", "Health Care Services, Department of")
        assert id1 == id2

    def test_different_values_produce_different_ids(self):
        id1 = _stable_id("Department Name", "Health Care Services, Department of")
        id2 = _stable_id("Department Name", "Public Health, Department of")
        assert id1 != id2

    def test_different_fields_same_value_produce_different_ids(self):
        id1 = _stable_id("Department Name", "Labor")
        id2 = _stable_id("Item Name", "Labor")
        assert id1 != id2

    def test_returns_hex_string(self):
        result = _stable_id("field", "value")
        assert len(result) == 16
        int(result, 16)  # raises ValueError if not valid hex
