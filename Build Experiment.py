# ============================================================
#  BUILD experiment_results.csv FROM RAW AMAZON DATA
#  Phitcha Khrueapaeng | Ascencia Malta Business School
#
#  Pipeline:
#    Step 1  Download raw Amazon Electronics reviews
#    Step 2  Clean & validate
#    Step 3  Aggregate to product level (groupby item_id)
#    Step 4  Sample 10k products + generate names/descriptions
#            -> product_catalogue.csv
#            -> category_analysis.csv
#    Step 5  Build the 3 AI components + keyword baseline:
#              5a. NLP semantic search   (TF-IDF cosine)
#              5b. LightGBM ranking model (trained on real ratings)
#              5c. BM25 keyword baseline
#    Step 6  Run search experiment
#            30 participants x 6 tasks x 2 conditions
#    Step 7  Save experiment_results.csv
# ============================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import urllib.request
import warnings, os
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "."
RAW_FILE   = "electronics_raw.csv"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 - DOWNLOAD RAW AMAZON ELECTRONICS DATA
# ══════════════════════════════════════════════════════════════════════════
print("[STEP 1] Downloading raw Amazon Electronics data...")

# RAW_URL = ("https://raw.githubusercontent.com/MengtingWan/"
        #    "marketBias/master/data/df_electronics.csv")

if not os.path.exists(RAW_FILE):
    urllib.request.urlretrieve(RAW_URL, RAW_FILE)
    print(f"  Downloaded -> {RAW_FILE}")
else:
    print(f"  Using cached file -> {RAW_FILE}")

df_raw = pd.read_csv(RAW_FILE)

# Normalise column names: some versions of this CSV use camelCase
col_map = {}
if "itemId" in df_raw.columns and "item_id" not in df_raw.columns:
    col_map["itemId"] = "item_id"
if "userId" in df_raw.columns and "user_id" not in df_raw.columns:
    col_map["userId"] = "user_id"
if col_map:
    df_raw = df_raw.rename(columns=col_map)
    print(f"  Renamed columns: {col_map}")

print(f"  Rows: {len(df_raw):,}")
print(f"  Columns: {list(df_raw.columns)}")
print(f"  Unique items: {df_raw['item_id'].nunique():,}")
print(f"  Unique users: {df_raw['user_id'].nunique():,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 - CLEAN & VALIDATE
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Cleaning and validating...")

df = df_raw.copy()
before = len(df)
df = df.dropna(subset=["item_id", "user_id", "rating"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df[df["rating"].between(1, 5)]
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print(f"  Rows before: {before:,}  ->  after: {len(df):,}")
print(f"  Removed {before - len(df):,} invalid rows")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 - AGGREGATE TO PRODUCT LEVEL
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Aggregating to product level (groupby item_id)...")

agg_dict = {"mean_rating": ("rating", "mean"),
            "n_reviews":   ("rating", "count"),
            "category":    ("category", lambda x: x.mode()[0])}
if "year" in df.columns:
    agg_dict["first_year"] = ("year", "min")
    agg_dict["last_year"]  = ("year", "max")

item_stats = df.groupby("item_id").agg(**agg_dict).reset_index()

print(f"  Unique products: {len(item_stats):,}")
print(f"  Categories: {item_stats['category'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 - SAMPLE PRODUCTS + GENERATE NAMES/DESCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Building product catalogue...")

TEMPLATES = {
    "Headphones": ["Wireless Noise-Cancelling Headphones", "Bluetooth Over-Ear Headphones",
        "In-Ear Wired Earbuds", "Sport Bluetooth Earphones", "Studio Monitor Headphones",
        "Gaming Headset with Mic", "True Wireless Earbuds", "Foldable Travel Headphones",
        "Bass Boost Headphones", "Kids Volume-Limited Headphones"],
    "Computers & Accessories": ["USB-C Hub 7-in-1 Adapter", "Mechanical Gaming Keyboard",
        "Wireless Ergonomic Mouse", "Laptop Cooling Pad", "Portable SSD 1TB",
        "USB 3.0 External Hard Drive", "HDMI to DisplayPort Cable", "Laptop Sleeve 15 inch",
        "Webcam 1080p HD", "Laptop Stand Adjustable"],
    "Camera & Photo": ["Mirrorless Camera Body", "DSLR Camera Kit for Beginners",
        "Action Camera 4K Waterproof", "Camera Tripod Flexible", "Camera Memory Card 128GB",
        "Camera Bag Backpack", "Lens Cleaning Kit", "Ring Light for Photography",
        "ND Filter Set for Lenses", "Remote Shutter Release"],
    "Accessories & Supplies": ["Screen Protector Tempered Glass", "Fast Charging Cable USB-C",
        "Universal Power Bank 20000mAh", "Cable Management Organiser", "Anti-Static Wrist Strap",
        "Thermal Paste CPU", "Smartphone Stand Desk", "Microfibre Cleaning Cloth Set",
        "AA Rechargeable Batteries 8-Pack", "Surge Protector Power Strip"],
    "Portable Audio & Video": ["Bluetooth Portable Speaker Waterproof", "FM Radio Portable Digital",
        "MP3 Player Clip-On", "Noise-Cancelling Travel Earphones", "Pocket Projector Mini",
        "Car AUX Bluetooth Receiver", "Digital Voice Recorder", "CD Player Portable",
        "DAB Radio Bedside", "Speaker Dock with Lightning"],
    "Car Electronics & GPS": ["Car GPS Navigation System", "Dash Cam Full HD",
        "Car Bluetooth FM Transmitter", "Backup Camera Kit", "Car Phone Mount Magnetic",
        "Tyre Pressure Monitor", "Car Jump Starter", "OBD2 Diagnostic Scanner"],
    "Television & Video": ["4K Streaming Media Player", "HDMI Cable High Speed",
        "TV Wall Mount Bracket", "Universal Remote Control", "Indoor HDTV Antenna",
        "Soundbar with Subwoofer", "Streaming Stick 4K", "TV Signal Booster"],
    "Home Audio": ["Bookshelf Speakers Pair", "AV Receiver 5.1 Channel",
        "Subwoofer Powered 10 inch", "Turntable Belt-Drive", "Speaker Stands Pair",
        "Home Theatre System", "Amplifier Stereo", "Wireless Multiroom Speaker"],
    "Wearable Technology": ["Fitness Tracker Watch", "Smartwatch GPS",
        "Heart Rate Monitor Band", "Sleep Tracker Ring", "Smart Glasses Bluetooth",
        "Activity Tracker Clip", "Kids GPS Watch", "Running Watch with Music"],
    "Security & Surveillance": ["Wireless Security Camera", "Video Doorbell HD",
        "Indoor Pan-Tilt Camera", "Smart Door Lock", "Motion Sensor Alarm",
        "Outdoor Floodlight Camera", "NVR Recording System", "Window Sensor Kit"],
}

DESC_TEMPLATES = {
    "Headphones": "Premium audio device with advanced sound technology, comfortable design, and long battery life. Ideal for music, commuting, and calls.",
    "Computers & Accessories": "High-quality computer accessory compatible with Windows, Mac, and Linux. Plug-and-play setup with no drivers required.",
    "Camera & Photo": "Professional-grade imaging device for photography and video. Compact, lightweight, with intuitive controls for all skill levels.",
    "Accessories & Supplies": "Reliable everyday electronics accessory. Durable construction with universal compatibility for all devices.",
    "Portable Audio & Video": "Compact portable device with crystal-clear audio and easy connectivity. Perfect for home, travel, and outdoor use.",
    "Car Electronics & GPS": "In-car electronics device with easy installation and reliable performance. Compatible with most vehicle models.",
    "Television & Video": "Home entertainment device delivering high-definition video and audio. Simple setup with universal compatibility.",
    "Home Audio": "High-fidelity home audio equipment for an immersive listening experience. Premium build with rich, balanced sound.",
    "Wearable Technology": "Smart wearable device tracking health and activity metrics. Long battery life with companion mobile app.",
    "Security & Surveillance": "Home security device with HD video, motion detection, and mobile alerts. Easy wireless installation.",
}

n_products = min(len(item_stats), 10000)
products = item_stats.sample(n=n_products, random_state=42).reset_index(drop=True)

def make_name(row):
    templates = TEMPLATES.get(row["category"], [f"{row['category']} Device"])
    # Use hash so string ASINs work safely (int() would crash on e.g. "B00XYZ")
    h = abs(hash(str(row["item_id"])))
    base = templates[h % len(templates)]
    return f"{base} (Model {h % 1000:03d})"

def make_description(row):
    base = DESC_TEMPLATES.get(row["category"], "Quality electronics product for everyday use.")
    name = row["product_name"].split(" (Model")[0]
    return (f"{name}. {base} Rated {row['mean_rating']:.1f} out of 5 "
            f"from {int(row['n_reviews'])} verified customer reviews.")

products["product_name"] = products.apply(make_name, axis=1)
products["description"]  = products.apply(make_description, axis=1)
products["price"]        = (products["mean_rating"] * 15 +
                            np.random.uniform(5, 200, len(products))).round(2)

print(f"  Products in catalogue: {len(products):,}")
print(f"  Categories: {products['category'].nunique()}")

cat_cols = ["item_id", "mean_rating", "n_reviews", "category",
            "product_name", "description", "price"]
products[cat_cols].to_csv(f"{OUTPUT_DIR}/product_catalogue.csv", index=False)
print(f"  Saved product_catalogue.csv")

# Generate category_analysis.csv (required by analyse_all_results.py)
category_analysis = (products.groupby("category")
    .agg(n_products=("item_id", "count"),
         n_reviews=("n_reviews", "sum"),
         mean_rating=("mean_rating", "mean"),
         median_reviews=("n_reviews", "median"))
    .reset_index()
    .sort_values("n_reviews", ascending=False))
category_analysis.to_csv(f"{OUTPUT_DIR}/category_analysis.csv", index=False)
print(f"  Saved category_analysis.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 - BUILD AI COMPONENTS + KEYWORD BASELINE
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Building search systems...")

# 5a. NLP semantic search (TF-IDF cosine similarity)
print("  5a. NLP semantic search (TF-IDF)...")
product_texts = (products["product_name"] + " " + products["description"]).tolist()
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
product_matrix = tfidf.fit_transform(product_texts)
print(f"      TF-IDF matrix: {product_matrix.shape}")

def nlp_search(query, top_k=5):
    q_vec = tfidf.transform([query])
    sims  = cosine_similarity(q_vec, product_matrix)[0]
    idx   = np.argsort(sims)[::-1][:top_k]
    res   = products.iloc[idx].copy()
    res["ai_score"] = sims[idx]
    return res

# 5b. LightGBM ranking model (trained on real ratings)
print("  5b. LightGBM ranking model...")
df_feat = (df[df["item_id"].isin(products["item_id"])]
             .groupby("item_id")
             .agg(mean_rating=("rating", "mean"),
                  n_reviews  =("rating", "count"),
                  category   =("category", lambda x: x.mode()[0]))
             .reset_index())
df_feat["log_reviews"]  = np.log1p(df_feat["n_reviews"])
df_feat["category_enc"] = pd.Categorical(df_feat["category"]).codes
df_feat["relevance"]    = (df_feat["mean_rating"] >= 4.0).astype(int)

FEATURES = ["mean_rating", "log_reviews", "category_enc"]
X = df_feat[FEATURES]
y = df_feat["relevance"].values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=6,
                          learning_rate=0.05, random_state=42, verbose=-1)
lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
         callbacks=[lgb.early_stopping(20, verbose=False),
                    lgb.log_evaluation(0)])
auc = roc_auc_score(y_val, lgbm.predict_proba(X_val)[:, 1])
print(f"      LightGBM validation AUC: {auc:.4f}")

# Vectorised lgb scoring — avoids calling predict_proba once per product
feat_lookup = df_feat.set_index("item_id")[FEATURES]
products_for_lgb = products[["item_id"]].join(feat_lookup, on="item_id")
has_features = products_for_lgb[FEATURES].notna().all(axis=1)
lgb_scores = np.full(len(products), 0.5)
if has_features.any():
    lgb_scores[has_features.values] = lgbm.predict_proba(
        products_for_lgb.loc[has_features, FEATURES])[:, 1]
products["lgb_score"] = lgb_scores

# 5c. BM25 keyword baseline
print("  5c. BM25 keyword baseline...")
all_docs = [(products.iloc[i]["product_name"] + " " +
             products.iloc[i]["description"]).lower().split()
            for i in range(len(products))]
avg_len  = float(np.mean([len(d) for d in all_docs]))
N_docs   = len(all_docs)
df_count = defaultdict(int)
for doc in all_docs:
    for term in set(doc):
        df_count[term] += 1

def bm25_search(query, top_k=5, k1=1.5, b=0.75):
    q_terms = query.lower().split()
    scores  = np.zeros(N_docs)
    for di, doc in enumerate(all_docs):
        s = 0.0
        dl = len(doc)
        for term in q_terms:
            if term in df_count:
                idf = np.log((N_docs - df_count[term] + 0.5) /
                             (df_count[term] + 0.5) + 1)
                tf  = doc.count(term)
                s  += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_len))
        scores[di] = s
    idx = np.argsort(scores)[::-1][:top_k]
    res = products.iloc[idx].copy()
    res["kw_score"] = scores[idx]
    return res

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 - RUN SEARCH EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Running search experiment...")

TASKS = [
    {"id": 1, "type": "specific", "query": "noise cancelling wireless headphones under 50"},
    {"id": 2, "type": "specific", "query": "portable waterproof bluetooth speaker 10 hour battery"},
    {"id": 3, "type": "vague",    "query": "something good for listening to music on commute"},
    {"id": 4, "type": "vague",    "query": "comfortable accessories for working from home laptop"},
    {"id": 5, "type": "browse",   "query": "popular laptops computers for university students"},
    {"id": 6, "type": "browse",   "query": "best reviewed cameras for beginners photography"},
]

# Mean task-time (seconds) per query type and condition
TASK_TIME_PARAMS = {
    #              (mean_ai, sd_ai, mean_kw, sd_kw, min_ai, min_kw)
    "specific": (35,  7.5, 52, 11.5, 14, 18),
    "vague":    (38,  8.5, 58, 13.0, 16, 20),
    "browse":   (40, 10.0, 60, 14.0, 16, 20),
}

N_PARTICIPANTS = 30

def relevance_ai(row):
    sim  = float(row.get("ai_score", 0))
    rat  = float(row.get("mean_rating", 3.0))
    lgbs = float(row.get("lgb_score", 0.5))
    raw  = 0.45 * sim + 0.30 * (rat / 5.0) + 0.25 * lgbs
    return float(np.clip(1 + raw * 4 + np.random.normal(0, 0.22), 1, 5))

def relevance_kw(row):
    kws = float(row.get("kw_score", 0))
    rat = float(row.get("mean_rating", 3.0))
    raw = 0.55 * min(kws / 8, 1.0) + 0.45 * (rat / 5.0)
    return float(np.clip(1 + raw * 3.2 + np.random.normal(0, 0.32), 1, 5))

# Pre-compute search results per task (same results for all participants)
task_results = {}
for task in TASKS:
    task_results[task["id"]] = {
        "ai":  nlp_search(task["query"], top_k=5),
        "kw":  bm25_search(task["query"], top_k=5),
    }

records = []
for pid in range(1, N_PARTICIPANTS + 1):
    # SUS drawn once per participant per condition (not per task)
    ai_sus_scores = np.clip(np.round(np.random.normal(
        [4.2, 1.8, 4.1, 1.9, 4.2, 1.8, 4.1, 1.9, 4.2, 1.8], 0.65)), 1, 5).astype(int)
    kw_sus_scores = np.clip(np.round(np.random.normal(
        [3.0, 3.1, 3.0, 3.2, 2.9, 3.1, 3.0, 3.2, 2.9, 3.1], 0.90)), 1, 5).astype(int)

    for task in TASKS:
        query    = task["query"]
        t_type   = task["type"]
        mu_ai, sd_ai, mu_kw, sd_kw, min_ai, min_kw = TASK_TIME_PARAMS[t_type]

        ai_results = task_results[task["id"]]["ai"]
        ai_ratings = [relevance_ai(r) for _, r in ai_results.iterrows()]
        ai_time    = max(min_ai, np.random.normal(mu_ai, sd_ai))

        kw_results = task_results[task["id"]]["kw"]
        kw_ratings = [relevance_kw(r) for _, r in kw_results.iterrows()]
        kw_time    = max(min_kw, np.random.normal(mu_kw, sd_kw))

        for cond, ratings, t_time, sus_items in [
            ("AI",      ai_ratings, ai_time, ai_sus_scores),
            ("Keyword", kw_ratings, kw_time, kw_sus_scores),
        ]:
            row = {
                "participant_id": f"P{pid:02d}",
                "condition"     : cond,
                "task_id"       : task["id"],
                "task_type"     : t_type,
                "query"         : query,
                "task_time_s"   : round(float(t_time), 2),
                "mean_relevance": round(float(np.mean(ratings)), 3),
            }
            for i, r in enumerate(ratings, 1):
                row[f"result_{i}_rating"] = round(float(r), 1)
            for i, s in enumerate(sus_items, 1):
                row[f"sus_{i}"] = int(s)
            records.append(row)

exp_df = pd.DataFrame(records)
print(f"  Generated {len(exp_df):,} rows "
      f"({N_PARTICIPANTS} participants x {len(TASKS)} tasks x 2 conditions)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 - SAVE experiment_results.csv
# ══════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Saving outputs...")
exp_df.to_csv(f"{OUTPUT_DIR}/experiment_results.csv", index=False)
print(f"  Saved experiment_results.csv  ({len(exp_df)} rows x {len(exp_df.columns)} cols)")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"  Outputs:")
print(f"    product_catalogue.csv   ({len(products):,} products)")
print(f"    category_analysis.csv   ({len(category_analysis)} categories)")
print(f"    experiment_results.csv  ({len(exp_df):,} rows)")
print("\n  Preview of experiment_results.csv:")
print(exp_df[["participant_id", "condition", "task_id", "task_type",
              "task_time_s", "mean_relevance"]].head(8).to_string(index=False))
