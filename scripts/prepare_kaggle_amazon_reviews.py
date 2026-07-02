from pathlib import Path
import pandas as pd
import sys


def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_category(text):
    text = str(text).lower()

    if "kindle" in text or "fire tablet" in text or "tablet" in text:
        return "Tablet / E-reader"
    if "fire tv" in text or "tv stick" in text:
        return "Streaming Device"
    if "echo" in text or "alexa" in text:
        return "Smart Speaker"
    if "vacuum" in text:
        return "Vacuum"
    if "blender" in text:
        return "Blender"
    if "coffee" in text:
        return "Coffee Maker"

    return "Consumer Product"


def sentiment_from_rating(rating):
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "mixed"
    return "positive"


def normalize_reviews(input_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    product_col = first_existing(df, ["name", "product_name", "ProductName", "product"])
    brand_col = first_existing(df, ["brand", "manufacturer", "manufacturer.name"])
    category_col = first_existing(df, ["primaryCategories", "categories", "category"])
    rating_col = first_existing(df, ["reviews.rating", "rating", "Score"])
    title_col = first_existing(df, ["reviews.title", "review_title", "Summary", "title"])
    text_col = first_existing(df, ["reviews.text", "review_text", "Text", "text"])
    date_col = first_existing(df, ["reviews.date", "review_date", "date", "Time"])
    id_col = first_existing(df, ["id", "reviews.id", "review_id"])

    if product_col is None or rating_col is None or text_col is None:
        raise ValueError(f"Missing required columns. Columns found: {list(df.columns)}")

    out = pd.DataFrame()
    out["review_id"] = df[id_col].astype(str) if id_col else [f"KAG{i+1:05d}" for i in range(len(df))]
    out["product_name"] = df[product_col].fillna("Unknown Product").astype(str)
    out["brand"] = df[brand_col].fillna("Unknown").astype(str) if brand_col else "Unknown"

    raw_category = df[category_col].fillna("").astype(str) if category_col else out["product_name"]
    out["category"] = [
        infer_category(f"{product} {category}")
        for product, category in zip(out["product_name"], raw_category)
    ]

    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    out["price"] = 0
    out["review_title"] = df[title_col].fillna("").astype(str) if title_col else ""
    out["review_text"] = df[text_col].fillna("").astype(str)

    if date_col:
        out["review_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        out["review_date"] = "2023-01-01"

    out["review_date"] = out["review_date"].fillna("2023-01-01")
    out = out.dropna(subset=["rating"])
    out = out[out["review_text"].str.len() > 20]
    out = out.drop_duplicates(subset=["product_name", "review_text"])

    out["sentiment_label"] = out["rating"].apply(sentiment_from_rating)
    out["feature_tags"] = "['General']"

    low_mid = out[out["rating"] <= 3]
    high = out[out["rating"] > 3]

    if len(low_mid) > 1200:
        low_mid = low_mid.sample(1200, random_state=7)

    if len(high) > 800:
        high = high.sample(800, random_state=7)

    final = pd.concat([low_mid, high], ignore_index=True)
    final = final.sample(frac=1, random_state=7).reset_index(drop=True)

    Path(output_path).parent.mkdir(exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"Saved {len(final)} cleaned real Amazon reviews to {output_path}")
    print("\nCategory counts:")
    print(final["category"].value_counts())
    print("\nSample rows:")
    print(final[["brand", "product_name", "category", "rating"]].head(20).to_string(index=False))


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/kaggle_amazon_reviews.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/real_amazon_reviews.csv"
    normalize_reviews(input_path, output_path)
