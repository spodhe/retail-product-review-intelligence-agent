import os
import re
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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

    if "brand" not in df.columns:
        return None

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
        "which products", "what products", "products have",
        "which product", "what product", "product has"
    ]
    return any(term in q for term in product_terms)


def remove_positive_no_complaint_reviews(df):
    bad_phrases = [
        "no complaints", "zero complaints", "no complaint",
        "no issues", "no issue", "no concerns", "zero concerns"
    ]

    text = (
        df["review_title"].fillna("") + " " +
        df["review_text"].fillna("")
    ).str.lower()

    mask = text.apply(lambda x: not any(phrase in x for phrase in bad_phrases))
    return df[mask].copy()


def apply_question_filters(df, question):
    temp = df.copy()

    detected_category = detect_category(question)
    detected_brand = detect_brand(question, temp)

    if detected_category and "category" in temp.columns:
        category_mask = temp["category"].astype(str).str.lower().eq(detected_category.lower())

        product_text = (
            temp["product_name"].fillna("") + " " +
            temp["review_text"].fillna("")
        ).str.lower()

        if detected_category == "Vacuum":
            text_mask = product_text.str.contains("vacuum|cleaner|suction|cordless", regex=True)
        elif detected_category == "Air Fryer":
            text_mask = product_text.str.contains("air fryer|fryer|basket|crispy", regex=True)
        elif detected_category == "Blender":
            text_mask = product_text.str.contains("blender|blend|smoothie|ice", regex=True)
        elif detected_category == "Coffee Maker":
            text_mask = product_text.str.contains("coffee|espresso|brew|keurig", regex=True)
        else:
            text_mask = False

        temp = temp[category_mask | text_mask]

    if detected_brand and "brand" in temp.columns:
        temp = temp[temp["brand"].astype(str).str.lower().eq(str(detected_brand).lower())]

    if is_complaint_query(question):
        complaint_filtered = temp[
            temp["sentiment_label"].isin(["negative", "mixed"]) |
            (pd.to_numeric(temp["rating"], errors="coerce") <= 3)
        ].copy()

        complaint_filtered = remove_positive_no_complaint_reviews(complaint_filtered)

        if not complaint_filtered.empty:
            temp = complaint_filtered

    return temp


def retrieve_reviews(df: pd.DataFrame, question: str, max_rows: int = 8):
    original_df = df.copy()
    temp = apply_question_filters(df, question)

    detected_category = detect_category(question)
    detected_brand = detect_brand(question, original_df)

    if temp.empty and (detected_category or detected_brand):
        parts = []
        if detected_brand:
            parts.append(str(detected_brand))
        if detected_category:
            parts.append(str(detected_category).lower())

        target = " ".join(parts).strip()

        return None, (
            f"I could not find matching reviews for **{target}** in the currently loaded dataset. "
            "Try uploading a dataset that includes that brand/category, or ask a broader question."
        )

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
        return None, (
            "I could not find matching complaint reviews. Try asking about a category, brand, product, "
            "or feature like durability, price/value, customer support, battery, suction, or noise."
        )

    return results, None


def summarize_themes(results):
    feature_rows = []

    for _, row in results.iterrows():
        for feature in row["feature_tags_list"]:
            feature_rows.append({
                "feature": feature,
                "rating": row["rating"]
            })

    if not feature_rows:
        return pd.DataFrame(columns=["feature", "mentions", "avg_rating"])

    feature_df = pd.DataFrame(feature_rows)

    return (
        feature_df.groupby("feature")
        .agg(
            mentions=("feature", "count"),
            avg_rating=("rating", "mean")
        )
        .reset_index()
        .sort_values(["mentions", "avg_rating"], ascending=[False, True])
        .head(5)
    )


def product_summary(results):
    if results.empty:
        return pd.DataFrame()

    return (
        results.groupby(["brand", "product_name", "category"])
        .agg(
            matching_reviews=("review_id", "count"),
            avg_rating=("rating", "mean")
        )
        .reset_index()
        .sort_values(["matching_reviews", "avg_rating"], ascending=[False, True])
        .head(6)
    )


def build_context(results, question):
    theme_summary = summarize_themes(results)
    prod_summary = product_summary(results)

    theme_lines = []
    for _, row in theme_summary.iterrows():
        theme_lines.append(
            f"- {pretty_feature(row['feature'])}: {int(row['mentions'])} mentions, "
            f"average rating {row['avg_rating']:.2f}"
        )

    product_lines = []
    for _, row in prod_summary.iterrows():
        display_name = clean_display_name(row["brand"], row["product_name"])
        product_lines.append(
            f"- {display_name} ({row['category']}): {int(row['matching_reviews'])} matching reviews, "
            f"average rating {row['avg_rating']:.2f}"
        )

    review_lines = []
    for i, (_, row) in enumerate(results.head(8).iterrows(), start=1):
        display_name = clean_display_name(row["brand"], row["product_name"])
        pretty_tags = [pretty_feature(tag) for tag in row["feature_tags_list"]]
        review_lines.append(
            f"[{i}] Product: {display_name}\n"
            f"Category: {row['category']}\n"
            f"Rating: {row['rating']}\n"
            f"Feature tags: {pretty_tags}\n"
            f"Review: {row['review_text']}"
        )

    return f"""
Business question:
{question}

Detected complaint themes:
{chr(10).join(theme_lines) if theme_lines else "No theme summary available."}

Product summary:
{chr(10).join(product_lines) if product_lines else "No product summary available."}

Evidence reviews:
{chr(10).join(review_lines)}
"""


def call_openai_llm(context):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    prompt = f"""
You are a consumer-products analytics assistant. Use only the review evidence below.
Answer like a business analyst for product, quality, marketing, and customer experience teams.

Rules:
- Do not invent facts outside the evidence.
- Start with a direct answer.
- Include top themes, affected products or brands, and recommended actions.
- Cite evidence using review numbers like [1], [2].
- Keep the response concise and business-oriented.

{context}
"""

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text


def call_anthropic_llm(context):
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return None

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    message = client.messages.create(
        model=model,
        max_tokens=900,
        system=(
            "You are a consumer-products analytics assistant. Use only the provided review evidence. "
            "Answer like a business analyst for product, quality, marketing, and customer experience teams. "
            "Do not invent facts."
        ),
        messages=[
            {
                "role": "user",
                "content": f"""
Use the evidence below to answer the business question.
Include top themes, affected products or brands, evidence references, and recommended actions.

{context}
"""
            }
        ]
    )

    return message.content[0].text


def deterministic_answer(results, question):
    theme_summary = summarize_themes(results)

    lines = []
    lines.append("### Retrieved-review answer")

    if not theme_summary.empty:
        lines.append("The top complaint themes in the matching reviews are:")

        for _, row in theme_summary.iterrows():
            lines.append(
                f"- **{pretty_feature(row['feature'])}**: "
                f"{int(row['mentions'])} mention(s), average rating **{row['avg_rating']:.2f}**"
            )

    if is_product_summary_query(question):
        lines.append("\n### Product summary")

        prod_summary = product_summary(results)

        for _, row in prod_summary.iterrows():
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
    if df.empty:
        return "No reviews are available for the selected filters."

    results, error = retrieve_reviews(df, question)

    if error:
        return error

    context = build_context(results, question)
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    try:
        if provider == "anthropic":
            llm_answer = call_anthropic_llm(context)
        else:
            llm_answer = call_openai_llm(context)

        if llm_answer:
            return (
                "### LLM-powered business answer\n\n"
                + llm_answer
                + "\n\n---\n"
                + "### Retrieved review evidence used by the LLM\n\n"
                + deterministic_answer(results, question)
            )

    except Exception as exc:
        return (
            "### LLM call failed\n\n"
            "The app retrieved matching reviews, but the LLM API call failed with this error:\n\n"
            f"`{exc}`\n\n"
            "Showing retrieved-review fallback instead.\n\n"
            + deterministic_answer(results, question)
        )

    return (
        "### LLM-ready retrieved-review answer\n\n"
        "No LLM API key is configured, so the app is using the local retrieval fallback. "
        "Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to enable LLM-powered answers.\n\n"
        + deterministic_answer(results, question)
    )
