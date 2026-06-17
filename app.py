import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import DATA_DIR, OPENAI_API_KEY
from src.ingest_reviews import load_reviews
from src.feature_analysis import feature_summary, top_pain_points, explode_features
from src.llm_summaries import generate_business_summary
from src.rag_chat import rag_answer

st.set_page_config(page_title="Retail Product Review Intelligence Agent", layout="wide")

st.title("Retail Product Review Intelligence Agent")
st.caption("AI + analytics dashboard for consumer product reviews")

def pretty_feature_name(feature):
    feature = str(feature)
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
    return mapping.get(feature, feature.replace("_", " ").title())


def priority_label(avg_rating, review_count):
    if avg_rating <= 2.0 and review_count >= 8:
        return "High Priority"
    elif avg_rating <= 2.7 and review_count >= 5:
        return "Medium Priority"
    return "Monitor"


uploaded = st.sidebar.file_uploader("Upload review CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample data", value=True)

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    temp_path = DATA_DIR / "_uploaded_reviews.csv"
    raw_df.to_csv(temp_path, index=False)
    df = load_reviews(temp_path)
elif use_sample:
    df = load_reviews(DATA_DIR / "sample_reviews.csv")
else:
    st.info("Upload a CSV or select sample data.")
    st.stop()

st.sidebar.header("Filters")
categories = sorted(df["category"].dropna().unique())
brands = sorted(df["brand"].dropna().unique())

selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
selected_brands = st.sidebar.multiselect("Brand", brands, default=brands)
rating_range = st.sidebar.slider("Rating range", 1.0, 5.0, (1.0, 5.0), 0.5)

filtered = df[
    df["category"].isin(selected_categories)
    & df["brand"].isin(selected_brands)
    & df["rating"].between(rating_range[0], rating_range[1])
].copy()

if filtered.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Reviews", len(filtered))
col2.metric("Avg rating", f"{filtered['rating'].mean():.2f}")
col3.metric("Brands", filtered["brand"].nunique())
col4.metric("Products", filtered["product_name"].nunique())

tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Pain Points",
    "AI Summary",
    "Ask the Data"
])

with tab1:
    st.subheader("Ratings by Brand")
    brand_rating = filtered.groupby("brand", as_index=False)["rating"].mean()
    fig = px.bar(brand_rating, x="brand", y="rating", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Mentions by Sentiment")
    summary = feature_summary(filtered)
    fig2 = px.bar(
        summary,
        x="feature_tags",
        y="review_count",
        color="sentiment_label",
        facet_col="category",
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Raw Reviews")
    st.dataframe(
        filtered[[
            "review_id", "product_name", "brand", "category", "rating",
            "price", "feature_tags", "sentiment_label", "review_text"
        ]],
        use_container_width=True
    )

with tab2:
    st.subheader("Top Pain Points")
    pain = top_pain_points(filtered, top_n=15)
    st.dataframe(pain, use_container_width=True)

    if not pain.empty:
        fig3 = px.bar(
            pain,
            x="feature_tags",
            y="review_count",
            color="brand",
            hover_data=["category", "avg_rating"]
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Recommended Business Actions")

        def action_for_feature(feature):
            feature = str(feature).lower()

            if "durability" in feature:
                return {
                    "team": "Product / Quality",
                    "action": "Investigate material quality, repeated-use failure points, warranty claims, and return reasons."
                }
            elif "battery" in feature:
                return {
                    "team": "Product / Marketing",
                    "action": "Compare advertised runtime against real usage patterns and clarify battery expectations in product messaging."
                }
            elif "noise" in feature:
                return {
                    "team": "Product / Marketing",
                    "action": "Benchmark noise levels against competitors and decide whether to adjust design priorities or product-page claims."
                }
            elif "cleaning" in feature:
                return {
                    "team": "Customer Experience / Product",
                    "action": "Improve cleaning instructions, redesign high-friction removable parts, or highlight easy-clean features more clearly."
                }
            elif "price" in feature or "value" in feature:
                return {
                    "team": "Marketing / Commercial Strategy",
                    "action": "Review price positioning, bundles, promotions, and value messaging against competitor alternatives."
                }
            elif "support" in feature:
                return {
                    "team": "Customer Experience",
                    "action": "Audit support response time, warranty clarity, replacement-part communication, and escalation paths."
                }
            elif "suction" in feature or "performance" in feature:
                return {
                    "team": "Product",
                    "action": "Validate performance claims against customer use cases and identify where expectations differ from real usage."
                }
            else:
                return {
                    "team": "Cross-functional",
                    "action": "Investigate this issue cluster with more review data, return data, and support-ticket analysis."
                }

        for _, row in pain.head(7).iterrows():
            recommendation = action_for_feature(row["feature_tags"])

            feature_name = pretty_feature_name(row["feature_tags"])
            priority = priority_label(row["avg_rating"], row["review_count"])

            st.markdown(
                f"""
**{priority}: {row['brand']} / {row['category']} — {feature_name}**  
Reviews affected: **{row['review_count']}**  
Average rating: **{row['avg_rating']:.2f}**  
Recommended owner: **{recommendation['team']}**  
Action: {recommendation['action']}
"""
            )

with tab3:
    st.subheader("AI-Generated Business Summary")
    st.write("This uses an LLM if your API key is configured. Otherwise it uses a fallback summary.")
    if st.button("Generate summary"):
        with st.spinner("Generating summary..."):
            summary_text = generate_business_summary(filtered, api_key=OPENAI_API_KEY)
        st.markdown(summary_text)

with tab4:
    st.subheader("Ask a business question")
    st.caption("Example: What are the top complaints about vacuums? Which brand has more durability complaints?")
    question = st.text_input("Question")
    if st.button("Ask") and question:
        with st.spinner("Searching reviews and generating answer..."):
            answer = rag_answer(filtered, question, api_key=OPENAI_API_KEY)
        st.markdown(answer)
