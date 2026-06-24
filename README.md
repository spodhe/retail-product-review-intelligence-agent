# Retail Product Review Intelligence & Marketplace Monitor

A SharkNinja-targeted AI and analytics portfolio project built with Python, Streamlit, pandas, Plotly, and scikit-learn.

## Project Overview

This project analyzes consumer product reviews and marketplace-style product metrics to help business teams identify customer pain points, prioritize product risks, and monitor competitive movement across appliance categories such as vacuums, air fryers, blenders, and coffee makers.

The app is designed for product, marketing, customer experience, and analytics teams that need to turn messy review and marketplace data into actionable business recommendations.

## Key Features

* Product review dashboard with brand, category, rating, and sentiment filters
* Feature-level complaint tagging across themes such as durability, price/value, suction/performance, battery, noise, customer support, and ease of cleaning
* Natural-language Q&A layer using TF-IDF retrieval to answer business questions with review evidence
* Product risk scoring based on review volume, average rating, and negative/mixed review share
* Trend tracking for review volume, ratings, and complaint themes over time
* Business action recommendations mapped to Product, Quality, Marketing, and Customer Experience teams
* API-ready marketplace monitor for product price, rating, review count, sales rank, and competitive movement

## Example Business Questions

* What are the top complaints about vacuums?
* What are the top complaints about Shark vacuums?
* Which products have durability issues?
* Which brands have price/value problems?
* Which products show the highest marketplace risk?
* Which competitors are gaining review traction?

## Tech Stack

* Python
* Streamlit
* pandas
* NumPy
* Plotly
* scikit-learn
* TF-IDF retrieval
* Rule-based feature tagging
* CSV-based data ingestion

## Data Note

The review intelligence workflow uses a clean sample consumer-product review dataset created to simulate realistic patterns across appliance categories. The marketplace monitor uses an API-ready sample data structure designed to represent fields that could be populated from product-monitoring APIs such as Keepa or similar marketplace data providers.

No private company data is used.

## How to Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Business Impact

This project demonstrates how AI-assisted analytics can help consumer product companies monitor customer feedback, detect recurring product issues, compare competitors, and translate review signals into actionable recommendations.
