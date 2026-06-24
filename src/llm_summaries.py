import os
import pandas as pd


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
        "noise": "Noise"
    }
    return mapping.get(str(feature), str(feature).replace("_", " ").title())


def owner_for_feature(feature):
    f = str(feature).lower()
    if "durability" in f:
        return "Product / Quality"
    if "support" in f:
        return "Customer Experience"
    if "price" in f or "value" in f:
        return "Marketing / Commercial Strategy"
    if "battery" in f or "noise" in f:
        return "Product / Marketing"
    if "cleaning" in f:
        return "Customer Experience / Product"
    if "suction" in f or "performance" in f:
        return "Product"
    return "Cross-functional"


def action_for_feature(feature):
    f = str(feature).lower()
    if "durability" in f:
        return "Investigate material quality, repeated-use failure points, warranty claims, and return reasons."
    if "support" in f:
        return "Audit support response times, warranty clarity, replacement-part communication, and escalation paths."
    if "price" in f or "value" in f:
        return "Review price positioning, bundles, promotions, and value messaging against competitors."
    if "battery" in f:
        return "Compare advertised runtime against real usage patterns and clarify battery expectations."
    if "noise" in f:
        return "Benchmark noise against competitors and review whether design or product-page claims need adjustment."
    if "cleaning" in f:
        return "Improve cleaning instructions, redesign high-friction removable parts, or highlight easy-clean features."
    if "suction" in f or "performance" in f:
        return "Validate performance claims against customer use cases and identify expectation gaps."
    return "Investigate this issue cluster with more review data, return data, and support-ticket analysis."


def build_review_context(df: pd.DataFrame, max_reviews: int = 20) -> str:
    sample = df.head(max_reviews)
    return "\n".join(
        f"Review ID: {row.get('review_id')} | Brand: {row.get('brand')} | "
        f"Product: {row.get('product_name')} | Rating: {row.get('rating')} | "
        f"Text: {row.get('review_text')}"
        for _, row in sample.iterrows()
    )


def fallback_summary(df: pd.DataFrame) -> str:
    from src.feature_analysis import top_pain_points

    if df.empty:
        return "No reviews available for the selected filters."

    pain = top_pain_points(df, top_n=8)

    total_reviews = len(df)
    avg_rating = df["rating"].mean()
    brand_count = df["brand"].nunique()
    product_count = df["product_name"].nunique()
    negative_share = df["sentiment_label"].isin(["negative", "mixed"]).mean()

    lines = []
    lines.append("## Executive Summary")
    lines.append(f"- Analyzed **{total_reviews} reviews** across **{brand_count} brands** and **{product_count} products**.")
    lines.append(f"- The average rating is **{avg_rating:.2f}**, with **{negative_share:.0%}** of reviews classified as negative or mixed.")

    if not pain.empty:
        top = pain.iloc[0]
        lines.append(
            f"- The highest-priority signal is **{top['brand']} / {top['category']} — {pretty_feature(top['feature_tags'])}**, "
            f"with **{int(top['review_count'])} affected reviews** and an average rating of **{top['avg_rating']:.2f}**."
        )

    lines.append("\n## Top Business Risks")
    for _, row in pain.head(5).iterrows():
        priority = "High" if row["avg_rating"] <= 2.0 and row["review_count"] >= 8 else "Medium"
        lines.append(
            f"- **{priority} Priority — {row['brand']} / {row['category']} / {pretty_feature(row['feature_tags'])}:** "
            f"{int(row['review_count'])} reviews affected, avg rating **{row['avg_rating']:.2f}**. "
            f"Owner: **{owner_for_feature(row['feature_tags'])}**."
        )

    lines.append("\n## Recommended Actions")
    for _, row in pain.head(5).iterrows():
        lines.append(
            f"- **{owner_for_feature(row['feature_tags'])} — {pretty_feature(row['feature_tags'])}:** "
            f"{action_for_feature(row['feature_tags'])}"
        )

    lines.append("\n## Data Quality Notes")
    lines.append("- Current conclusions should be validated with larger public or internal review data.")
    lines.append("- Feature tagging is rule-based, so future versions should validate tags against labeled examples or LLM-assisted classification.")
    lines.append("- Stronger inputs would include returns, support tickets, warranty claims, and product-level sales volume.")

    lines.append("\n## Suggested Next Steps")
    lines.append("- Use public Amazon Reviews data for a stronger dataset.")
    lines.append("- Add TF-IDF or vector retrieval for stronger natural-language Q&A.")
    lines.append("- Deploy the app publicly using Streamlit Community Cloud.")

    return "\n".join(lines)


def generate_business_summary(df: pd.DataFrame, api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return fallback_summary(df)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a consumer product analytics assistant. Use only the supplied review data."),
            ("human", """
Analyze these product reviews.

Return:
## Executive Summary
## Top Business Risks
## Recommended Actions
## Data Quality Notes

Reviews:
{context}
""")
        ])

        result = (prompt | llm).invoke({"context": build_review_context(df)})
        return result.content

    except Exception as e:
        return fallback_summary(df) + f"\n\nLLM summary failed, using fallback. Error: {e}"
