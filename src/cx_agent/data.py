"""FABSA dataset loader"""

import os
from dotenv import load_dotenv
import ast
import pandas as pd
from datasets import load_dataset
load_dotenv()

# Valid filter values
VALID_INDUSTRIES = [
    "Banking", "Consulting", "Fashion", "Groceries",
    "Information Technology", "Price Comparison", "Ride Hailing",
    "Streaming", "Trading", "Travel Booking",
]

VALID_DATA_SOURCES = ["Trustpilot", "Google Play", "Apple Store"]

VALID_PARENT_ASPECTS = [
    "company-brand", "logistics-rides", "online-experience",
    "purchase-booking-experience", "staff-support", "value",
    "account-management",
]

VALID_CHILD_ASPECTS = [
    "account-access", "app-website", "attitude-of-staff", "competitor",
    "discounts-promotions", "ease-of-use", "email", "general-satisfaction",
    "phone", "price-value-for-money", "reviews", "speed",
]

VALID_SENTIMENTS = ["positive", "negative", "neutral"]

SENTIMENT_MAP = {"-1": "negative", "0": "neutral", "1": "positive"}

def load_fabsa() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FABSA (all splits) and return (df_reviews, df_exploded).

    df_reviews:  one row per review
    df_exploded: one row per aspect+sentiment label
    """
    ds = load_dataset("jordiclive/FABSA", token=os.getenv("HF_TOKEN"))
    df = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)

    # Parse label_codes
    df["label_codes_parsed"] = df["label_codes"].apply(ast.literal_eval)
    df["num_labels"] = df["label_codes_parsed"].apply(len)
    df["word_count"] = df["text"].str.split().apply(len)

    # Explode into one row per label
    df_exploded = df.explode("label_codes_parsed").reset_index(drop=True)
    df_exploded["parent_aspect"] = df_exploded["label_codes_parsed"].apply(
        lambda x: x.split(".")[0]
    )
    df_exploded["child_aspect"] = df_exploded["label_codes_parsed"].apply(
        lambda x: x.split(".")[1]
    )
    df_exploded["sentiment"] = df_exploded["label_codes_parsed"].apply(
        lambda x: SENTIMENT_MAP[x.split(".")[2]]
    )

    return df, df_exploded