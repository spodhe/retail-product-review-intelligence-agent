import re
import pandas as pd

FEATURE_KEYWORDS = {
    "suction/performance": [
        "suction", "performance", "cleans well", "powerful", "weak", "boost",
        "struggles", "pet hair", "carpet", "debris", "dust", "crumbs",
        "multiple passes", "picks up"
    ],
    "battery": [
        "battery", "charge", "charging", "cordless", "runtime", "boost mode"
    ],
    "noise": [
        "loud", "noisy", "quiet", "fan", "motor noise", "noise", "high-power setting"
    ],
    "durability": [
        "durable", "durability", "peeling", "broke", "broken", "leaking",
        "sturdy", "coating", "cracked", "crack", "brush roll", "hose connection",
        "failure", "reliability", "repeated use"
    ],
    "ease_of_cleaning": [
        "clean", "cleanup", "empty", "dust bin", "basket", "filter",
        "rinse", "hair gets tangled", "tangled", "remove", "removable parts"
    ],
    "price/value": [
        "price", "expensive", "cheap", "affordable", "value", "overpriced",
        "competitors", "hard to justify"
    ],
    "customer_support": [
        "support", "customer service", "warranty", "replacement", "refund",
        "respond", "clear answer", "replacement parts"
    ],
    "design/usability": [
        "heavy", "lightweight", "easy to use", "controls", "maneuver",
        "simple", "handle", "button", "attachments", "store", "steer",
        "furniture", "awkward"
    ],
    "odor/smell": [
        "smell", "odor", "plastic smell", "burning smell"
    ]
}

NEGATIVE_WORDS = [
    "bad", "terrible", "annoying", "loud", "weak", "shorter", "peeling", "leaking",
    "heavy", "expensive", "struggles", "problem", "issue", "not as easy", "too long"
]

POSITIVE_WORDS = [
    "great", "excellent", "easy", "strong", "powerful", "crispy", "durable", "sturdy",
    "simple", "good", "fast", "evenly", "affordable"
]

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def tag_features(text: str) -> list[str]:
    text_norm = normalize_text(text)
    tags = []
    for feature, keywords in FEATURE_KEYWORDS.items():
        if any(keyword in text_norm for keyword in keywords):
            tags.append(feature)
    return tags or ["general"]

def sentiment_label(row) -> str:
    text = normalize_text(str(row.get("review_title", "")) + " " + str(row.get("review_text", "")))
    rating = float(row.get("rating", 0))
    pos_hits = sum(word in text for word in POSITIVE_WORDS)
    neg_hits = sum(word in text for word in NEGATIVE_WORDS)

    if rating <= 2 or neg_hits > pos_hits:
        return "negative"
    if rating >= 4 and pos_hits >= neg_hits:
        return "positive"
    return "mixed"

def enrich_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = df["review_title"].fillna("") + " " + df["review_text"].fillna("")
    df["feature_tags"] = df["combined_text"].apply(tag_features)
    df["sentiment_label"] = df.apply(sentiment_label, axis=1)
    return df

def explode_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = enrich_reviews(df)
    return enriched.explode("feature_tags")

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_features(df)
    summary = (
        exploded
        .groupby(["category", "brand", "feature_tags", "sentiment_label"])
        .size()
        .reset_index(name="review_count")
    )
    return summary

def top_pain_points(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    exploded = explode_features(df)
    neg = exploded[exploded["sentiment_label"].isin(["negative", "mixed"])]
    out = (
        neg.groupby(["category", "brand", "feature_tags"])
        .agg(review_count=("review_id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values(["review_count", "avg_rating"], ascending=[False, True])
        .head(top_n)
    )
    return out
