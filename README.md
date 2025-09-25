# Amazon-reviews-analysis
Topic modeling of 20,000 negative Amazon reviews using BERTopic and OpenAI for meaningful topic labeling.

# Amazon Reviews Topic Modeling with BERTopic & OpenAI

This project analyzes **20,000 negative Amazon reviews** (Score = 1) using 
**Natural Language Processing, BERTopic, and OpenAI LLMs** to extract meaningful 
topics from customer complaints.

## 🚀 Features
- Extracted 20,000 reviews from SQLite database
- Preprocessed text with NLTK
- Applied BERTopic with PCA for clustering
- Used OpenAI GPT models for human-readable topic labels
- Exported CSVs and interactive HTML visualizations

## 📂 Project Structure
- `bertopic_analysis.py` — Main script
- `requirements.txt` — Dependencies
- `outputs/` — CSV + HTML results

## 🔧 Installation
```bash
git clone https://github.com/<Hesamdgh>/amazon-reviews-analysis.git
cd amazon-reviews-analysis
pip install -r requirements.txt
