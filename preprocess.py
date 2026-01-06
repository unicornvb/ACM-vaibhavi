# src/preprocess.py
import pandas as pd
import json


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    else:
        return pd.read_csv(path)


def combine_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["description", "input_description", "output_description"]

    for c in cols:
        if c not in df.columns:
            df[c] = ""

    df["combined_text"] = (
        df["description"].fillna("") + "\n" +
        df["input_description"].fillna("") + "\n" +
        df["output_description"].fillna("")
    )

    return df
