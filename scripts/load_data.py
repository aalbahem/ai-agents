"""Load procurement CSV data into MongoDB."""

import pathlib

import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "procurement"
COLLECTION_NAME = "purchases"
CSV_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data" / "lpca" / "PURCHASE ORDER DATA EXTRACT 2012-2015_0.csv"
)


def load_data() -> None:
    print(f"Reading CSV from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"Read {len(df):,} rows")

    # --- type conversions ---
    # Dates: MM/DD/YYYY → datetime
    for col in ["Creation Date", "Purchase Date"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    # Prices: strip $ and , → float
    for col in ["Unit Price", "Total Price"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quantity → float
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Code columns: keep as strings
    code_cols = ["Supplier Code", "Normalized UNSPSC", "Class", "Family", "Segment"]
    for col in code_cols:
        df[col] = df[col].astype(str).replace("nan", None)

    # Convert to records and replace NaN/NaT with None for MongoDB
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None

    # --- insert into MongoDB ---
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Drop existing data for idempotent re-runs
    collection.drop()
    print("Inserting documents ...")

    batch_size = 10_000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        collection.insert_many(batch)
        print(f"  inserted {min(i + batch_size, len(records)):,} / {len(records):,}")

    print(f"Total documents: {collection.count_documents({})}")

    # --- create indexes ---
    indexes = [
        "Creation Date",
        "Fiscal Year",
        "Department Name",
        "Supplier Name",
        "Acquisition Type",
        "Item Name",
        "Total Price",
    ]
    for field in indexes:
        collection.create_index(field)
        print(f"  created index on '{field}'")

    print("Done.")
    client.close()


if __name__ == "__main__":
    load_data()
