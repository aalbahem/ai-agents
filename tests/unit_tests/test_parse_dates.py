"""Level 1: Unit tests for _parse_dates() from src/tools/procurement.py."""

from datetime import datetime

from tools.procurement import _parse_dates


class TestParseDatesISODatetime:
    def test_iso_datetime_string(self):
        result = _parse_dates("2013-01-01T00:00:00")
        assert result == datetime(2013, 1, 1, 0, 0, 0)

    def test_iso_datetime_with_time(self):
        result = _parse_dates("2014-06-15T14:30:00")
        assert result == datetime(2014, 6, 15, 14, 30, 0)


class TestParseDatesISODate:
    def test_iso_date_only(self):
        result = _parse_dates("2013-01-01")
        assert result == datetime(2013, 1, 1, 0, 0, 0)

    def test_iso_date_mid_year(self):
        result = _parse_dates("2015-07-20")
        assert result == datetime(2015, 7, 20, 0, 0, 0)


class TestParseDatesPassthrough:
    def test_plain_string(self):
        assert _parse_dates("hello") == "hello"

    def test_fiscal_year_string(self):
        assert _parse_dates("2013-2014") == "2013-2014"

    def test_number(self):
        assert _parse_dates(42) == 42

    def test_float(self):
        assert _parse_dates(3.14) == 3.14

    def test_none(self):
        assert _parse_dates(None) is None

    def test_boolean(self):
        assert _parse_dates(True) is True


class TestParseDatesNestedDicts:
    def test_date_range_query(self):
        obj = {"$gte": "2013-01-01T00:00:00", "$lt": "2014-01-01T00:00:00"}
        result = _parse_dates(obj)
        assert result == {
            "$gte": datetime(2013, 1, 1),
            "$lt": datetime(2014, 1, 1),
        }

    def test_match_stage_with_dates(self):
        obj = {
            "$match": {
                "Creation Date": {
                    "$gte": "2013-01-01T00:00:00",
                    "$lt": "2014-01-01T00:00:00",
                }
            }
        }
        result = _parse_dates(obj)
        assert result["$match"]["Creation Date"]["$gte"] == datetime(2013, 1, 1)
        assert result["$match"]["Creation Date"]["$lt"] == datetime(2014, 1, 1)

    def test_mixed_dict_values(self):
        obj = {"date": "2013-06-01", "name": "hello", "count": 5}
        result = _parse_dates(obj)
        assert result == {
            "date": datetime(2013, 6, 1),
            "name": "hello",
            "count": 5,
        }


class TestParseDatesLists:
    def test_list_of_dates(self):
        result = _parse_dates(["2013-01-01", "2014-01-01"])
        assert result == [datetime(2013, 1, 1), datetime(2014, 1, 1)]

    def test_list_of_mixed_types(self):
        result = _parse_dates(["2013-01-01", "hello", 42])
        assert result == [datetime(2013, 1, 1), "hello", 42]


class TestParseDatesAggregationPipeline:
    def test_full_pipeline(self):
        pipeline = [
            {
                "$match": {
                    "Creation Date": {
                        "$gte": "2013-01-01T00:00:00",
                        "$lt": "2014-01-01T00:00:00",
                    }
                }
            },
            {"$count": "total"},
        ]
        result = _parse_dates(pipeline)
        assert result[0]["$match"]["Creation Date"]["$gte"] == datetime(2013, 1, 1)
        assert result[0]["$match"]["Creation Date"]["$lt"] == datetime(2014, 1, 1)
        assert result[1] == {"$count": "total"}

    def test_pipeline_without_dates(self):
        pipeline = [
            {"$group": {"_id": "$Fiscal Year", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        result = _parse_dates(pipeline)
        assert result == pipeline
