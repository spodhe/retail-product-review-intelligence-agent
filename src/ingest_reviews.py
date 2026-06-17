import pandas as pd
from pathlib import Path
from src.feature_analysis import enrich_reviews

REQUIRED_COLUMNS = [
    "review_id", "product_name", "brand", "category", "rating", "price",
    "review_title", "review_text", "review_date"
]

def load_reviews(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_id", "review_text"])
    return enrich_reviews(df)

def save_enriched_reviews(input_path: str | Path, output_path: str | Path) -> None:
    df = load_reviews(input_path)
    # Save feature tags as pipe-separated string for CSV readability.
    df["feature_tags"] = df["feature_tags"].apply(lambda tags: "|".join(tags))
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = Path("data/sample_reviews.csv")
    output_path = Path("data/enriched_reviews.csv")
    save_enriched_reviews(input_path, output_path)
    print(f"Saved enriched reviews to {output_path}")
