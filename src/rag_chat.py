import os
import re
import ast
import pandas as pd


def clean_display_name(brand, product):
    brand = str(brand)
    product = str(product)

    if product.lower().startswith(brand.lower()):
        return product

    return f"{brand} {product}"


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


def simple_keyword_answer(df: pd.DataFrame, question: str, max_rows: int = 6) -> str:
    """
    Fallback answer without vector DB or LLM.
    Handles plural/singular matching and summarizes top complaint themes.
    """

    def normalize(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def singularize(token: str) -> str:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    stopwords = {
        "what", "are", "the", "top", "about", "which", "have", "has",
        "with", "from", "that", "this", "they", "them", "most", "more",
        "less", "than", "into", "data", "complaints", "complaint",
        "customer", "customers", "saying", "say", "product", "products",
        "review", "reviews", "issues", "issue"
    }

    question_norm = normalize(question)
    raw_terms = question_norm.split()
    terms = [singularize(term) for term in raw_terms if term not in stopwords and len(term) > 2]

    temp = df.copy()
    temp["feature_tags_list"] = temp["feature_tags"].apply(parse_feature_tags)
    temp["feature_tags_text"] = temp["feature_tags_list"].apply(lambda tags: " ".join(tags))

    temp["search_text"] = (
        temp["product_name"].fillna("") + " " +
        temp["brand"].fillna("") + " " +
        temp["category"].fillna("") + " " +
        temp["review_title"].fillna("") + " " +
        temp["review_text"].fillna("") + " " +
        temp["feature_tags_text"].fillna("") + " " +
        temp["sentiment_label"].fillna("")
    ).apply(normalize)

    def score_row(text: str) -> int:
        text_tokens = set(singularize(token) for token in text.split())
        score = 0

        for term in terms:
            if term in text_tokens or term in text:
                score += 1

        return score

    temp["match_score"] = temp["search_text"].apply(score_row)

    results = temp[temp["match_score"] > 0].sort_values(
        ["match_score", "rating"],
        ascending=[False, True]
    )

    if results.empty:
        return (
            "I could not find matching reviews. Try asking about a brand, category, "
            "or feature like battery, noise, suction, durability, price, Shark, Ninja, Dyson, or air fryer."
        )

    negative_or_mixed = results[results["sentiment_label"].isin(["negative", "mixed"])]

    if negative_or_mixed.empty:
        negative_or_mixed = results

    # Aggregate complaint themes
    feature_rows = []

    for _, row in negative_or_mixed.iterrows():
        for feature in row["feature_tags_list"]:
            feature_rows.append({
                "feature": feature,
                "brand": row["brand"],
                "category": row["category"],
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
                f"- **{row['feature']}**: {int(row['mentions'])} mention(s), "
                f"average rating **{row['avg_rating']:.2f}**"
            )

    lines.append("\n### Evidence examples")

    examples = negative_or_mixed.head(max_rows)

    for _, row in examples.iterrows():
        display_name = clean_display_name(row["brand"], row["product_name"])
        features = row.get("feature_tags_list", [])

        lines.append(
            f"- **{display_name}** ({row['rating']} stars): "
            f"{row['review_text']}  \n"
            f"  Feature tags: `{features}`"
        )

    lines.append("\n### Business interpretation")
    lines.append(
        "The strongest business signals are the complaint themes that combine low ratings "
        "with repeated mentions. These should be prioritized because they likely affect customer satisfaction, "
        "returns, reviews, and brand perception."
    )

    return "\n".join(lines)


def rag_answer(df: pd.DataFrame, question: str, api_key: str | None = None) -> str:
    """
    Lightweight RAG. Uses LangChain + FAISS + OpenAI if API key exists.
    Falls back to keyword search if not.
    """

    api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        return simple_keyword_answer(df, question)

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate

        docs = []

        for _, row in df.iterrows():
            content = (
                f"Review ID: {row['review_id']}\n"
                f"Brand: {row['brand']}\n"
                f"Product: {row['product_name']}\n"
                f"Category: {row['category']}\n"
                f"Rating: {row['rating']}\n"
                f"Price: {row['price']}\n"
                f"Review: {row['review_text']}"
            )

            docs.append(
                Document(
                    page_content=content,
                    metadata={"review_id": row["review_id"]}
                )
            )

        embeddings = OpenAIEmbeddings(api_key=api_key)
        vectorstore = FAISS.from_documents(docs, embeddings)
        retrieved = vectorstore.similarity_search(question, k=6)
        context = "\n\n".join(doc.page_content for doc in retrieved)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You answer business questions using only the supplied customer review context. "
                "Do not invent facts. Mention review IDs when useful."
            ),
            (
                "human",
                "Question: {question}\n\nReview context:\n{context}"
            )
        ])

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
        result = (prompt | llm).invoke({"question": question, "context": context})
        return result.content

    except Exception as e:
        return simple_keyword_answer(df, question) + f"\n\nRAG failed, using fallback. Error: {e}"
