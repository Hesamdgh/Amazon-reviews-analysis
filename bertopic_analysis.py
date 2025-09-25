from umap import UMAP
from hdbscan import HDBSCAN
import sqlite3
import pandas as pd
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from sklearn.decomposition import PCA
import time
import ast
import re
from openai import OpenAI
import openai  # for exception handling

# --- Setup ---
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# OpenAI client (ensure OPENAI_API_KEY is set in your environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to your SQLite file
DB_PATH = "/Users/hesamghanbari/.cache/kagglehub/datasets/joychakraborty2000/amazon-customers-data/versions/1/database.sqlite"
OUTPUT_DIR = "bertopic_output_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load reviews ---
print("Loading reviews...")
conn = sqlite3.connect(DB_PATH)
query = "SELECT Score, Text FROM Reviews;"
reviews = pd.read_sql_query(query, conn)
conn.close()
print(f"Loaded {len(reviews)} reviews.")

# --- 2. Filter Score = 1 and sample 20,000 reviews ---
reviews = reviews[reviews["Score"] == 1]

if len(reviews) > 20000:
    reviews = reviews.sample(20000, random_state=42)

print(f"Score=1 sampled reviews: {len(reviews)}")

# --- 3. Clean text ---
def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", str(text).lower())
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 2
              and w not in ["product", "amazon", "like", "good"]]
    return " ".join(tokens)

reviews["CleanText"] = reviews["Text"].astype(str).apply(clean_text)

# --- 4. Run BERTopic ---
docs = reviews["CleanText"].dropna().tolist()
print("Fitting BERTopic on 20,000 Score==1 reviews...")

umap_model = UMAP(
    n_neighbors=15,   # controls local vs global structure
    n_components=5,   # dimensionality
    min_dist=0.0,     # tighter clusters
    metric="cosine",
    random_state=42
)


topic_model = BERTopic(
    language="english",
    verbose=True,
    umap_model=umap_model,
    min_topic_size=15,   # instead of 50
    calculate_probabilities=True
)



hdbscan_model = HDBSCAN(
    min_cluster_size=15,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

topic_model = BERTopic(
    language="english",
    verbose=True,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True
)

# Reassign outliers to closest topics
topics, probs = topic_model.fit_transform(docs)
topics = topic_model.reduce_outliers(docs, topics)

# --- 5. Extract topic info ---
topic_info = topic_model.get_topic_info()

# --- Helpers for labeling ---
def parse_keywords(cell):
    if isinstance(cell, (list, tuple)):
        # Already a list of keywords
        return [str(x) for x in cell]
    if isinstance(cell, str):
        # Try parsing string that looks like a Python list
        try:
            obj = ast.literal_eval(cell)
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj]
        except Exception:
            return re.findall(r"[a-zA-Z0-9']+", cell)[:10]
    if pd.isna(cell):
        return []
    return [str(cell)]

def make_prompt(batch_keywords):
    header = (
        "You are an assistant that turns clusters of keywords into short, "
        "human-readable topic labels (2–4 words). Return a numbered list only.\n\n"
    )
    examples = (
        "Example:\n"
        "Keywords: shipping, late, delivery, arrived\n"
        "Label: Delivery Delays\n\n"
        "Keywords: broken, defect, stopped, damaged\n"
        "Label: Product Quality Issues\n\n"
    )
    body = "Now label the following clusters:\n"
    for i, kwlist in enumerate(batch_keywords, start=1):
        body += f"{i}. {', '.join(kwlist)}\n"
    return header + examples + body

def call_openai(prompt):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limit hit, sleeping {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print("OpenAI error:", e)
            time.sleep(2 ** attempt)
    return ""

def parse_labels(response_text, expected_n):
    labels = []
    for line in response_text.splitlines():
        m = re.match(r'^\s*(\d+)[\).\-\s]+(.+)$', line.strip())
        if m:
            labels.append(m.group(2).strip())
    if len(labels) < expected_n:
        labels += [""] * (expected_n - len(labels))
    return labels[:expected_n]

# --- 6. Generate labels in batches ---
all_labels = []
batch_size = 15
representations = topic_info["Representation"].fillna("").apply(parse_keywords).tolist()

for i in range(0, len(representations), batch_size):
    batch = representations[i:i+batch_size]
    if not any(batch):
        all_labels.extend([""] * len(batch))
        continue
    prompt = make_prompt(batch)
    response_text = call_openai(prompt)
    labels = parse_labels(response_text, len(batch)) if response_text else [""] * len(batch)
    all_labels.extend(labels)
    time.sleep(0.2)

topic_info["Topic_Label"] = all_labels

# --- 7. Save outputs ---
csv_path = f"{OUTPUT_DIR}/topics_sample_reviews_labeled.csv"
topic_info.to_csv(csv_path, index=False)
print(f"Saved topics table with labels → {csv_path}")

topic_model.visualize_barchart().write_html(f"{OUTPUT_DIR}/topics_barchart_score1.html")
topic_model.visualize_topics().write_html(f"{OUTPUT_DIR}/topics_map_score1.html")
topic_model.visualize_hierarchy().write_html(f"{OUTPUT_DIR}/topics_hierarchy_score1.html")

print("\n✅ Done. Check the 'bertopic_output_2' folder for CSV + HTML files.")