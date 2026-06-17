import os
from typing import Optional
import pandas as pd

def build_review_context(df: pd.DataFrame, max_reviews: int = 20) -> str:
    sample = df.head(max_reviews)
    lines = []
    for _, row in sample.iterrows():
        lines.append(
            f"Review ID: {row.get('review_id')} | Brand: {row.get('brand')} | "
            f"Product: {row.get('product_name')} | Rating: {row.get('rating')} | "
            f"Text: {row.get('review_text')}"
        )
    return "\n".join(lines)

def fallback_summary(df: pd.DataFrame) -> str:
    from src.feature_analysis import top_pain_points
    pain = top_pain_points(df, top_n=5)
    if pain.empty:
        return "No major pain points found in the filtered reviews."
    bullets = []
    for _, row in pain.iterrows():
        bullets.append(
            f"- {row['brand']} / {row['category']}: {row['feature_tags']} appeared in "
            f"{row['review_count']} negative or mixed review(s), avg rating {row['avg_rating']:.1f}."
        )
    return "Top detected pain points:\n" + "\n".join(bullets)

def generate_business_summary(df: pd.DataFrame, api_key: Optional[str] = None) -> str:
    """
    Uses an LLM if an API key is available. Otherwise returns a rule-based summary.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return fallback_summary(df)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a consumer product analytics assistant. "
             "Use only the supplied review data. Be specific, business-oriented, and concise."),
            ("human",
             """
Analyze these product reviews.

Return:
1. Top 5 customer pain points
2. Which brands/products are most affected
3. Likely business impact
4. Recommended actions for Product, Marketing, and Customer Experience teams
5. 3 risks or data gaps

Reviews:
{context}
""")
        ])

        chain = prompt | llm
        result = chain.invoke({"context": build_review_context(df)})
        return result.content

    except Exception as e:
        return fallback_summary(df) + f"\n\nLLM summary failed, using fallback. Error: {e}"
