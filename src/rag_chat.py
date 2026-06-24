import os
import re
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def pretty_feature(feature):
    mapping = {
        "customer_support": "Customer Support",
        "ease_of_cleaning": "Ease of Cleaning",
        "price/value": "Price / Value",
        "suction/performance": "Suction / Performance",
        "design/usability": "Design / Usability",
        "odor/smell": "Odor / Smell",
        "durability": "Durability",
        "battery": "Battery",
        "noise": "Noise",
        "general": "General"
    }
    return mapping.get(str(feature), str(feature).replace("_", " ").title())


def clean_display_name(brand, product):
    brand = str(brand)
    product = str(product)
    return product if product.lower().startswith(brand.lower()) else f"{brand} {product}"


def parse_feature_tags(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        if "|" in value:
            return [item.strip() for item in value.split("|") if item.strip()]

        return [value]

    return ["general"]


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_category(question):
    q = normalize(question)

    if "vacuum" in q or "vacuums" in q or "cleaner" in q or "cleaners" in q:
        return "Vacuum"
    if "air fryer" in q or "air fryers" in q:
        return "Air Fryer"
    if "blender" in q or "blenders" in q:
        return "Blender"
    if "coffee" in q or "espresso" in q or "keurig" in q:
        return "Coffee Maker"

    return None


def detect_brand(question, df):
    q = normalize(question)
    brands = sorted(df["brand"].dropna().astype(str).unique(), key=len, reverse=True)

    for brand in brands:
        if normalize(brand) in q:
            return brand

    return None


def is_complaint_query(question):
    q = normalize(question)
    complaint_terms = [
        "complaint", "complaints", "problem", "problems", "issue", "issues",
        "bad", "negative", "concern", "concerns", "risk", "risks", "fix",
        "prioritize", "pain point", "pain points"
    ]
    return any(term in q for term in complaint_terms)


def is_product_summary_query(question):
    q = normalize(question)
    product_terms = [
        "which products",
        "what products",
        "products have",
        "which product",
        "what product",
        "product has"
    ]
    return any(term in q for term in product_terms)


def remove_positive_no_complaint_reviews(df):
    bad_phrases = [
        "no complaints",
        "zero complaints",
        "no complaint",
        "no issues",
        "no issue",
        "no concerns",
        "zero concerns"
    ]

    text = (
        df["review_title"].fillna("") + " " +
        df["review_text"].fillna("")
    ).str.lower()

    mask = text.apply(lambda x: not any(phrase in x for phrase in bad_phrases))
    return df[mask].copy()


def apply_question_filters(df, question):
    temp = df.copy()

    category = detect_category(question)
    brand = detect_brand(question, temp)

    if category:
        category_mask = temp["category"].astype(str).str.lower().eq(category.lower())

        product_text = (
            temp["product_name"].fillna("") + " " +
            temp["review_text"].fillna("")
        ).str.lower()

        if category == "Vacuum":
            text_mask = product_text.str.contains("vacuum|cleaner|suction|cordless", regex=True)
        elif category == "Air Fryer":
            text_mask = product_text.str.contains("air fryer|fryer|basket|crispy", regex=True)
        elif category == "Blender":
            text_mask = product_text.str.contains("blender|blend|smoothie|ice", regex=True)
        elif category == "Coffee Maker":
            text_mask = product_text.str.contains("coffee|espresso|brew|keurig", regex=True)
        else:
            text_mask = False

        temp = temp[category_mask | text_mask]

    if brand:
        temp = temp[temp["brand"].astype(str).str.lower().eq(str(brand).lower())]

    if is_complaint_query(question):
        complaint_filtered = temp[
            temp["sentiment_label"].isin(["negative", "mixed"]) |
            (pd.to_numeric(temp["rating"], errors="coerce") <= 3)
        ].copy()

        complaint_filtered = remove_positive_no_complaint_reviews(complaint_filtered)

        if not complaint_filtered.empty:
            temp = complaint_filtered

    return temp


def tfidf_answer(df: pd.DataFrame, question: str, max_rows: int = 8) -> str:
    if df.empty:
        return "No reviews are available for the selected filters."

    original_df = df.copy()
    temp = apply_question_filters(df, question)

    if temp.empty:
        temp = original_df.copy()

    temp["feature_tags_list"] = temp["feature_tags"].apply(parse_feature_tags)
    temp["feature_tags_text"] = temp["feature_tags_list"].apply(lambda x: " ".join(x))

    temp["search_text"] = (
        temp["brand"].fillna("") + " " +
        temp["product_name"].fillna("") + " " +
        temp["category"].fillna("") + " " +
        temp["review_title"].fillna("") + " " +
        temp["review_text"].fillna("") + " " +
        temp["feature_tags_text"].fillna("") + " " +
        temp["sentiment_label"].fillna("")
    )

    corpus = temp["search_text"].tolist() + [question]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    matrix = vectorizer.fit_transform(corpus)
    review_vectors = matrix[:-1]
    question_vector = matrix[-1]

    temp["similarity"] = cosine_similarity(question_vector, review_vectors).flatten()

    results = temp.sort_values(
        ["similarity", "rating"],
        ascending=[False, True]
    ).head(max_rows)

    if results.empty or results["similarity"].max() <= 0:
        results = temp.sort_values("rating", ascending=True).head(max_rows)

    if is_complaint_query(question):
        results = remove_positive_no_complaint_reviews(results)

    if results.empty:
        return (
            "I could not find matching complaint reviews. Try asking about a category, brand, product, "
            "or feature like durability, price/value, customer support, battery, suction, or noise."
        )

    feature_rows = []

    for _, row in results.iterrows():
        for feature in row["feature_tags_list"]:
            feature_rows.append({
                "feature": feature,
                "rating": row["rating"]
            })

    feature_df = pd.DataFrame(feature_rows)

    lines = []
    lines.append("### Review-based answer")

    if not feature_df.empty:
        theme_summary = (
            feature_df.groupby("feature")
            .agg(
                mentions=("feature", "count"),
                avg_rating=("rating", "mean")
            )
            .reset_index()
            .sort_values(["mentions", "avg_rating"], ascending=[False, True])
            .head(5)
        )

        lines.append("The top complaint themes in the matching reviews are:")

        for _, row in theme_summary.iterrows():
            lines.append(
                f"- **{pretty_feature(row['feature'])}**: "
                f"{int(row['mentions'])} mention(s), average rating **{row['avg_rating']:.2f}**"
            )

    if is_product_summary_query(question):
        lines.append("\n### Product summary")

        product_summary = (
            results.groupby(["brand", "product_name", "category"])
            .agg(
                matching_reviews=("review_id", "count"),
                avg_rating=("rating", "mean")
            )
            .reset_index()
            .sort_values(["matching_reviews", "avg_rating"], ascending=[False, True])
            .head(6)
        )

        for _, row in product_summary.iterrows():
            display_name = clean_display_name(row["brand"], row["product_name"])
            lines.append(
                f"- **{display_name}** ({row['category']}): "
                f"{int(row['matching_reviews'])} matching review(s), "
                f"average rating **{row['avg_rating']:.2f}**"
            )

    lines.append("\n### Evidence examples")

    for _, row in results.head(5).iterrows():
        display_name = clean_display_name(row["brand"], row["product_name"])
        pretty_tags = [pretty_feature(tag) for tag in row["feature_tags_list"]]

        lines.append(
            f"- **{display_name}** ({row['rating']} stars): {row['review_text']}  \n"
            f"  Feature tags: `{pretty_tags}`"
        )

    lines.append("\n### Business interpretation")
    lines.append(
        "The highest-risk themes are the ones that combine repeated mentions with low ratings. "
        "These should be prioritized because they may affect satisfaction, returns, reviews, and brand perception."
    )

    return "\n".join(lines)


def rag_answer(df: pd.DataFrame, question: str, api_key: str | None = None) -> str:
    return tfidf_answer(df, question)
