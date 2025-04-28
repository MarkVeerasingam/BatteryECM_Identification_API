import pandas as pd
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["BatteryData"]
collection = db["ECM_LUT"]

def insert_csv_to_mongodb(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"No data found in {csv_path}")
            return

        # Extract metadata
        battery_label = str(df["battery_label"].iloc[0])
        cycle = int(df["cycle"].iloc[0])

        # Convert data to Python-native types
        data_rows = df.drop(columns=["battery_label", "cycle"]).map(
            lambda x: x.item() if hasattr(x, "item") else x
        ).to_dict(orient="records")

        document = {
            "battery_label": battery_label,
            "cycle": cycle,
            "data": data_rows
        }

        # Replace or insert the document
        collection.replace_one(
            {"battery_label": battery_label, "cycle": cycle},
            document,
            upsert=True
        )

        print(f"Saved full ECM LUT for {battery_label} cycle {cycle} to MongoDB.")

    except Exception as e:
        print(f"Failed to insert CSV to MongoDB: {e}")