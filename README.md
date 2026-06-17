# Retail Product Review Intelligence Agent

A SharkNinja-targeted AI + analytics portfolio project.

## What this project does

This app analyzes consumer product reviews and helps a business user answer questions like:

- What are customers complaining about most?
- Which product features drive negative sentiment?
- How do brands compare across suction, noise, durability, price, battery, and ease of cleaning?
- What actions should product, marketing, or CX teams take next?
- Can a business user ask questions in natural language and get grounded answers from review data?

## Recommended resume bullet

Built a consumer product review intelligence agent using Python, SQL, LLM prompting, and Streamlit to classify customer pain points, summarize recurring product issues, and generate business recommendations across appliance categories.

## Folder structure

```text
data/
  sample_reviews.csv
sql/
  schema.sql
src/
  config.py
  feature_analysis.py
  llm_summaries.py
  rag_chat.py
  ingest_reviews.py
app.py
requirements.txt
.env.example
```

## MVP build order

1. Run the dashboard locally using the sample data.
2. Replace sample data with a real dataset.
3. Add LLM summaries.
4. Add RAG chatbot.
5. Add Supabase/Postgres storage.
6. Deploy on Streamlit Community Cloud.
7. Add project link and GitHub link to your resume.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Mac
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Notes

- The app works without an LLM key using simple rule-based analysis.
- Add an LLM API key later for stronger summaries and chatbot answers.
- Keep your API key out of GitHub.
