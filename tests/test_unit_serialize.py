"""Level 1: Unit tests for _serialize() from app/db.py."""

from datetime import datetime

from bson import ObjectId

from app.db import _serialize


class TestSerializeObjectId:
    def test_objectid_to_string(self):
        oid = ObjectId("507f1f77bcf86cd799439011")
        result = _serialize({"_id": oid})
        assert result["_id"] == "507f1f77bcf86cd799439011"
        assert isinstance(result["_id"], str)


class TestSerializeDatetime:
    def test_datetime_to_iso(self):
        dt = datetime(2013, 8, 27, 0, 0, 0)
        result = _serialize({"Creation Date": dt})
        assert result["Creation Date"] == "2013-08-27T00:00:00"

    def test_datetime_with_time(self):
        dt = datetime(2014, 6, 15, 14, 30, 45)
        result = _serialize({"ts": dt})
        assert result["ts"] == "2014-06-15T14:30:45"


class TestSerializePassthrough:
    def test_string_passthrough(self):
        result = _serialize({"name": "Pitney Bowes"})
        assert result["name"] == "Pitney Bowes"

    def test_number_passthrough(self):
        result = _serialize({"Total Price": 675.00})
        assert result["Total Price"] == 675.00

    def test_int_passthrough(self):
        result = _serialize({"count": 42})
        assert result["count"] == 42

    def test_none_passthrough(self):
        result = _serialize({"Purchase Date": None})
        assert result["Purchase Date"] is None


class TestSerializeMixedDocument:
    def test_full_document(self):
        doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "Creation Date": datetime(2013, 8, 27),
            "Supplier Name": "Pitney Bowes",
            "Total Price": 675.00,
            "Quantity": 4.5,
            "Purchase Date": None,
            "Fiscal Year": "2013-2014",
        }
        result = _serialize(doc)
        assert result == {
            "_id": "507f1f77bcf86cd799439011",
            "Creation Date": "2013-08-27T00:00:00",
            "Supplier Name": "Pitney Bowes",
            "Total Price": 675.00,
            "Quantity": 4.5,
            "Purchase Date": None,
            "Fiscal Year": "2013-2014",
        }
