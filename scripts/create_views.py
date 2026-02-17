"""Create MongoDB views for order-level aggregation.

The raw `purchases` collection stores one row per LINE ITEM. These views
pre-aggregate to the order level so that simple COUNT queries return the
correct number of purchase orders rather than line items.

A unique purchase order is identified by (Department Name, Purchase Order Number)
since PO numbers are NOT unique across departments.

Run after load_data.py:
    python scripts/create_views.py
"""

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "procurement"

VIEW_NAME = "purchase_orders"
SOURCE_COLLECTION = "purchases"

PIPELINE = [
    {
        "$group": {
            "_id": {
                "Department Name": "$Department Name",
                "Purchase Order Number": "$Purchase Order Number",
            },
            "Department Name": {"$first": "$Department Name"},
            "Purchase Order Number": {"$first": "$Purchase Order Number"},
            "Fiscal Year": {"$first": "$Fiscal Year"},
            "Creation Date": {"$min": "$Creation Date"},
            "Purchase Date": {"$min": "$Purchase Date"},
            "Supplier Name": {"$first": "$Supplier Name"},
            "Supplier Code": {"$first": "$Supplier Code"},
            "Acquisition Type": {"$first": "$Acquisition Type"},
            "Acquisition Method": {"$first": "$Acquisition Method"},
            "CalCard": {"$first": "$CalCard"},
            "LPA Number": {"$first": "$LPA Number"},
            "total_line_items": {"$sum": 1},
            "total_spent": {"$sum": "$Total Price"},
        }
    },
    {
        "$project": {
            "_id": 0,
            "Department Name": 1,
            "Purchase Order Number": 1,
            "Fiscal Year": 1,
            "Creation Date": 1,
            "Purchase Date": 1,
            "Supplier Name": 1,
            "Supplier Code": 1,
            "Acquisition Type": 1,
            "Acquisition Method": 1,
            "CalCard": 1,
            "LPA Number": 1,
            "total_line_items": 1,
            "total_spent": 1,
        }
    },
]


def create_views() -> None:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Drop existing view if present (idempotent)
    if VIEW_NAME in db.list_collection_names():
        db.drop_collection(VIEW_NAME)
        print(f"Dropped existing view '{VIEW_NAME}'")

    db.create_collection(VIEW_NAME, viewOn=SOURCE_COLLECTION, pipeline=PIPELINE)
    print(f"Created view '{VIEW_NAME}' on '{SOURCE_COLLECTION}'")

    # Quick sanity check
    raw_count = db[SOURCE_COLLECTION].count_documents({})
    view_count = db[VIEW_NAME].count_documents({})
    print(f"  Raw line items: {raw_count:,}")
    print(f"  Distinct orders: {view_count:,}")
    print(f"  Ratio: {raw_count / view_count:.1f} line items per order (average)")

    client.close()
    print("Done.")


if __name__ == "__main__":
    create_views()
