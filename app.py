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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Pain Points",
    "AI Summary",
    "Ask the Data",
    "Trend Tracking",
    "Product Risk",
    "Live Product Monitor"
])

with tab1:
    st.subheader("Ratings by Brand")
    brand_rating = filtered.groupby("brand", as_index=False)["rating"].mean()
    fig = px.bar(brand_rating, x="brand", y="rating", text_auto=".2f")
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Raw Reviews")
    st.dataframe(
        filtered[[
            "review_id", "product_name", "brand", "category", "rating",
            "price", "feature_tags", "sentiment_label", "review_text"
        ]],
        width="stretch"
    )

with tab2:
    st.subheader("Top Pain Points")
    pain = top_pain_points(filtered, top_n=15)
    st.dataframe(pain, width="stretch")

    if not pain.empty:
        fig3 = px.bar(
            pain,
            x="feature_tags",
            y="review_count",
            color="brand",
            hover_data=["category", "avg_rating"]
        )
        st.plotly_chart(fig3, width="stretch")

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


with tab5:
    st.subheader("Product-Level Trend Tracking")
    st.caption("Tracks review volume, average rating, complaint themes, and product risk over time.")

    trend_df = filtered.copy()
    trend_df["review_date"] = pd.to_datetime(trend_df["review_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["review_date"])

    if trend_df.empty:
        st.warning("No valid review dates found for trend tracking.")
    else:
        trend_df["review_month"] = trend_df["review_date"].dt.to_period("M").dt.to_timestamp()

        st.markdown("### Review Volume Over Time")

        monthly_volume = (
            trend_df.groupby(["review_month", "category"])
            .size()
            .reset_index(name="review_count")
        )

        fig_volume = px.line(
            monthly_volume,
            x="review_month",
            y="review_count",
            color="category",
            markers=True
        )

        st.plotly_chart(fig_volume, width="stretch")

        st.markdown("### Average Rating Over Time")

        monthly_rating = (
            trend_df.groupby(["review_month", "category"])
            .agg(avg_rating=("rating", "mean"))
            .reset_index()
        )

        fig_rating = px.line(
            monthly_rating,
            x="review_month",
            y="avg_rating",
            color="category",
            markers=True
        )

        st.plotly_chart(fig_rating, width="stretch")

        st.markdown("### Product Risk Score")

        product_risk = (
            trend_df.groupby(["brand", "product_name", "category"])
            .agg(
                review_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
                negative_or_mixed_share=(
                    "sentiment_label",
                    lambda x: x.isin(["negative", "mixed"]).mean()
                )
            )
            .reset_index()
        )

        product_risk["risk_score"] = (
            product_risk["review_count"]
            * product_risk["negative_or_mixed_share"]
            * (5 - product_risk["avg_rating"])
        )

        product_risk = product_risk.sort_values("risk_score", ascending=False)

        st.dataframe(
            product_risk,
            width="stretch"
        )

        st.markdown("### Complaint Theme Trends")

        exploded_trends = explode_features(trend_df)

        top_features = (
            exploded_trends.groupby("feature_tags")
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
            .head(6)["feature_tags"]
            .tolist()
        )

        feature_trends = exploded_trends[
            exploded_trends["feature_tags"].isin(top_features)
        ]

        monthly_features = (
            feature_trends.groupby(["review_month", "feature_tags"])
            .size()
            .reset_index(name="mentions")
        )

        fig_features = px.line(
            monthly_features,
            x="review_month",
            y="mentions",
            color="feature_tags",
            markers=True
        )

        st.plotly_chart(fig_features, width="stretch")

        st.markdown("### How to Interpret This")
        st.markdown(
            """
- **Rising review volume** can indicate growing customer attention or issue visibility.
- **Declining average rating** can signal product quality, support, or expectation gaps.
- **High product risk score** means a product has many reviews, a high negative/mixed share, and lower ratings.
- **Complaint theme trends** help teams see whether issues like durability, price/value, or support are getting worse over time.
"""
        )


with tab6:
    st.subheader("Product Risk Dashboard")
    st.caption("Identifies products with the strongest risk signal based on review volume, rating severity, and negative/mixed review share.")

    risk_df = filtered.copy()

    if risk_df.empty:
        st.warning("No data available for product risk analysis.")
    else:
        product_risk = (
            risk_df.groupby(["brand", "product_name", "category"])
            .agg(
                review_count=("review_id", "count"),
                avg_rating=("rating", "mean"),
                negative_mixed_share=(
                    "sentiment_label",
                    lambda x: x.isin(["negative", "mixed"]).mean()
                )
            )
            .reset_index()
        )

        product_risk["risk_score"] = (
            product_risk["review_count"]
            * product_risk["negative_mixed_share"]
            * (5 - product_risk["avg_rating"])
        )

        product_risk["priority"] = product_risk["risk_score"].apply(
            lambda x: "High" if x >= product_risk["risk_score"].quantile(0.75)
            else ("Medium" if x >= product_risk["risk_score"].quantile(0.40) else "Monitor")
        )

        product_risk = product_risk.sort_values("risk_score", ascending=False)

        st.markdown("### Highest-Risk Products")
        st.dataframe(product_risk, width="stretch")

        fig_risk = px.scatter(
            product_risk,
            x="avg_rating",
            y="negative_mixed_share",
            size="review_count",
            color="priority",
            hover_data=["brand", "product_name", "category", "risk_score"],
            title="Product Risk: Rating vs Negative/Mixed Review Share"
        )
        st.plotly_chart(fig_risk, width="stretch")

        top = product_risk.iloc[0]

        st.markdown("### Executive Risk Takeaway")
        st.markdown(
            f"""
The highest-risk product signal is **{top['brand']} / {top['product_name']}** in **{top['category']}**.

- Review count: **{int(top['review_count'])}**
- Average rating: **{top['avg_rating']:.2f}**
- Negative/mixed review share: **{top['negative_mixed_share']:.0%}**
- Risk score: **{top['risk_score']:.2f}**

This product should be reviewed first because it combines customer volume, lower satisfaction, and repeated negative or mixed feedback.
"""
        )


with tab7:
    st.subheader("Live Product Monitor")
    st.caption(
        "API-ready marketplace monitoring layer for product price, rating, review count, "
        "sales-rank movement, and competitive signals."
    )

    live_path = DATA_DIR / "live_product_monitor_sample.csv"

    if not live_path.exists():
        st.warning("Missing data/live_product_monitor_sample.csv. Run the setup command to generate it.")
    else:
        live_df = pd.read_csv(live_path)
        live_df["snapshot_date"] = pd.to_datetime(live_df["snapshot_date"])

        latest_date = live_df["snapshot_date"].max()
        latest = live_df[live_df["snapshot_date"] == latest_date].copy()

        st.info(
            "Demo note: this sample file is structured like output from a marketplace-data API. "
            "In production, the same table could be populated from a product-monitoring API."
        )

        monitor_categories = sorted(latest["category"].dropna().unique())
        monitor_brands = sorted(latest["brand"].dropna().unique())

        mcol1, mcol2 = st.columns(2)
        selected_monitor_categories = mcol1.multiselect(
            "Monitor category",
            monitor_categories,
            default=monitor_categories
        )
        selected_monitor_brands = mcol2.multiselect(
            "Monitor brand",
            monitor_brands,
            default=monitor_brands
        )

        monitor = latest[
            latest["category"].isin(selected_monitor_categories)
            & latest["brand"].isin(selected_monitor_brands)
        ].copy()

        if monitor.empty:
            st.warning("No monitored products match the selected filters.")
        else:
            # Competitive score: higher means stronger marketplace position.
            monitor["market_strength_score"] = (
                monitor["current_rating"] * 20
                + monitor["review_count_change_30d"] / 100
                - monitor["sales_rank"] / 5
            ).round(2)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Monitored Products", len(monitor))
            k2.metric("Avg Rating", f"{monitor['current_rating'].mean():.2f}")
            k3.metric("Total Reviews", f"{int(monitor['review_count'].sum()):,}")
            k4.metric("Avg 30D Price Change", f"${monitor['price_change_30d'].mean():.2f}")

            st.subheader("Latest Marketplace Snapshot")
            st.dataframe(
                monitor[[
                    "asin", "brand", "product_name", "category",
                    "current_price", "list_price", "current_rating",
                    "review_count", "sales_rank", "price_change_30d",
                    "review_count_change_30d", "market_strength_score"
                ]].sort_values("market_strength_score", ascending=False),
                width="stretch"
            )

            st.subheader("Price Position by Product")
            price_fig = px.bar(
                monitor.sort_values("current_price", ascending=False),
                x="product_name",
                y="current_price",
                color="brand",
                hover_data=["category", "current_rating", "review_count", "sales_rank"],
                text_auto=".2f"
            )
            st.plotly_chart(price_fig, width="stretch")

            st.subheader("30-Day Review Count Growth")
            review_growth_fig = px.bar(
                monitor.sort_values("review_count_change_30d", ascending=False),
                x="product_name",
                y="review_count_change_30d",
                color="brand",
                hover_data=["category", "current_rating", "sales_rank"]
            )
            st.plotly_chart(review_growth_fig, width="stretch")

            st.subheader("Competitive Position Map")
            scatter_fig = px.scatter(
                monitor,
                x="current_rating",
                y="sales_rank",
                size="review_count",
                color="brand",
                hover_name="product_name",
                hover_data=["category", "current_price", "review_count_change_30d"],
            )
            scatter_fig.update_yaxes(autorange="reversed", title="Sales Rank, lower is better")
            st.plotly_chart(scatter_fig, width="stretch")

            strongest = monitor.sort_values("market_strength_score", ascending=False).iloc[0]
            fastest = monitor.sort_values("review_count_change_30d", ascending=False).iloc[0]
            biggest_price_drop = monitor.sort_values("price_change_30d").iloc[0]

            st.subheader("Executive Marketplace Takeaway")
            st.markdown(
                f"""
- **Strongest current marketplace signal:** {strongest['brand']} {strongest['product_name']} with a market strength score of **{strongest['market_strength_score']:.2f}**.
- **Fastest review-count growth:** {fastest['brand']} {fastest['product_name']} gained **{int(fastest['review_count_change_30d']):,}** reviews over the monitored period.
- **Largest price drop:** {biggest_price_drop['brand']} {biggest_price_drop['product_name']} changed by **${biggest_price_drop['price_change_30d']:.2f}** versus list price.
- **Business use case:** this tab connects review intelligence with marketplace movement, helping teams monitor competitor pricing, customer traction, and category-level product risk.
"""
            )
