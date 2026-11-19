"""
Phishing URL detection: Random Forest vs Char-BiLSTM vs DistilBERT
on the PhiUSIIL dataset (matches Section 3.3 of the paper).

SETUP
    pip install pandas numpy scikit-learn matplotlib tensorflow torch transformers
    Download the dataset from the UCI repository (PhiUSIIL Phishing URL Dataset, id 967)
    and place PhiUSIIL_Phishing_URL_Dataset.csv next to this script.

USAGE
    python phishing_comparison.py                       # all three models, full data
    python phishing_comparison.py --models rf bilstm    # subset of models
    python phishing_comparison.py --sample 20000        # quick smoke test

OUTPUT (./results)
    results_table.csv, confusion_matrices.png, roc_curves.png, misclassified_<model>.csv

NOTE ON LABELS
    In PhiUSIIL, label 1 = legitimate and 0 = phishing. This script flips it so that
    phishing = 1 (the positive class). Then FPR = FP / (FP + TN) is the rate at which
    legitimate URLs are wrongly flagged as phishing.
"""
import argparse
import math
import os
import re
import time
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

SEED = 42
RESULTS_DIR = "results"
N_LATENCY = 300  # URLs used to measure single-URL inference time


# ----------------------------------------------------------------------------
# 1. Data loading and preprocessing (Section 3.3.1)
# ----------------------------------------------------------------------------
def load_data(path, sample=None):
    df = pd.read_csv(path, usecols=["URL", "label"])
    df = df.drop_duplicates(subset="URL")
    df["url"] = (df["URL"].str.lower()
                 .str.replace(r"^https?://", "", regex=True))
    df = df.drop_duplicates(subset="url")  # duplicates created by normalisation
    df["y"] = 1 - df["label"].astype(int)  # phishing = 1
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=SEED)
    return df[["url", "y"]].reset_index(drop=True)


def split_data(df):
    train, temp = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["y"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Lexical features for the Random Forest (20 features)
# ----------------------------------------------------------------------------
KEYWORDS = ("login", "verify", "secure", "account", "update", "bank", "signin", "confirm")
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "co", "in", "uk", "de", "io"}
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "dots", "hyphens", "digits", "digit_ratio",
    "letter_ratio", "special_chars", "at_signs", "slashes", "question_marks", "equals",
    "ampersands", "percents", "has_ip", "subdomains", "common_tld", "entropy", "keywords",
]


def entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def lexical_features(url):
    p = urlparse("//" + url)
    host = (p.hostname or "")
    parts = host.split(".")
    n = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    return [
        len(url), len(host), len(p.path), url.count("."), url.count("-"), digits, digits / n,
        letters / n, sum(not ch.isalnum() for ch in url), url.count("@"), url.count("/"),
        url.count("?"), url.count("="), url.count("&"), url.count("%"),
        int(bool(IP_RE.match(host))), max(len(parts) - 2, 0),
        int(parts[-1] in COMMON_TLDS) if parts else 0, entropy(url),
        sum(k in url for k in KEYWORDS),
    ]


def feature_matrix(urls):
    return np.array([lexical_features(u) for u in urls], dtype=np.float32)


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def compute_metrics(name, y_true, prob, ms_per_url):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "FPR": fp / (fp + tn),
        "ms/URL": ms_per_url,
    }, pred


def time_single(predict_one, urls):
    """Mean latency (ms) when scoring URLs one at a time, including preprocessing."""
    predict_one(urls[0])  # warm-up
    start = time.perf_counter()
    for u in urls:
        predict_one(u)
    return (time.perf_counter() - start) / len(urls) * 1000


# ----------------------------------------------------------------------------
# 4. Model 1: Random Forest
# ----------------------------------------------------------------------------
def run_rf(train, val, test):
    print("\n[M1] Random Forest")
    Xtr, Xte = feature_matrix(train.url), feature_matrix(test.url)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, train.y)
    prob = rf.predict_proba(Xte)[:, 1]
    ms = time_single(lambda u: rf.predict_proba(feature_matrix([u]))[:, 1], list(test.url[:N_LATENCY]))
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])[:5]
    print("Top features:", ", ".join(f"{k}={v:.3f}" for k, v in imp))
    return prob, ms


# ----------------------------------------------------------------------------
# 5. Model 2: Character-level BiLSTM
# ----------------------------------------------------------------------------
def run_bilstm(train, val, test, max_len=200, epochs=5, batch=256):
    print("\n[M2] Char-BiLSTM")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    from tensorflow.keras import layers, models, callbacks

    vocab = {ch: i + 2 for i, ch in enumerate(sorted(set("".join(train.url))))}  # 0 pad, 1 unk

    def encode(urls):
        out = np.zeros((len(urls), max_len), dtype=np.int32)
        for i, u in enumerate(urls):
            ids = [vocab.get(c, 1) for c in u[:max_len]]
            out[i, :len(ids)] = ids
        return out

    Xtr, Xva, Xte = encode(train.url), encode(val.url), encode(test.url)
    model = models.Sequential([
        layers.Embedding(len(vocab) + 2, 64, mask_zero=True),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, train.y, validation_data=(Xva, val.y), epochs=epochs, batch_size=batch,
              callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2)
    prob = model.predict(Xte, batch_size=512, verbose=0).ravel()
    ms = time_single(lambda u: model(encode([u]), training=False),
                     list(test.url[:N_LATENCY]))
    return prob, ms


# ----------------------------------------------------------------------------
# 6. Model 3: DistilBERT (custom PyTorch loop, version-independent)
# ----------------------------------------------------------------------------
def run_distilbert(train, val, test, epochs=2, lr=2e-5, batch=32, max_len=128):
    print("\n[M3] DistilBERT")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)

    def enc(urls):
        return tok(list(urls), truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ytr = torch.tensor(train.y.values)
    n = len(train)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))

    def predict_probs(urls, bs=128):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(urls), bs):
                b = {k: v.to(dev) for k, v in enc(urls[i:i + bs]).items()}
                out.append(torch.softmax(model(**b).logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(SEED + ep).permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = {k: v.to(dev) for k, v in enc(train.url.values[idx]).items()}
            loss = model(**b, labels=ytr[idx].to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
        vp = predict_probs(list(val.url))
        vacc = accuracy_score(val.y, vp >= 0.5)
        print(f"epoch {ep + 1}/{epochs}  train loss {total / math.ceil(n / batch):.4f}  val acc {vacc:.4f}")

    prob = predict_probs(list(test.url))
    ms = time_single(lambda u: predict_probs([u]), list(test.url[:N_LATENCY]))
    print(f"Device: {dev}")
    return prob, ms


# ----------------------------------------------------------------------------
# 7. Plots and error analysis (Section 3.4)
# ----------------------------------------------------------------------------
def make_plots(y, probs, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(preds), figsize=(4.5 * len(preds), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y, pred)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Phish"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Phish"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, prob):.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=200)
    plt.close()


def save_errors(name, test, pred):
    err = test.assign(pred=pred)
    err = err[err.y != err.pred].copy()
    err["error_type"] = np.where(err.pred == 1, "false_positive", "false_negative")
    fname = name.lower().replace(" ", "_").replace("-", "_")
    err.to_csv(f"{RESULTS_DIR}/misclassified_{fname}.csv", index=False)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="PhiUSIIL_Phishing_URL_Dataset.csv")
    ap.add_argument("--models", nargs="+", default=["rf", "bilstm", "distilbert"],
                    choices=["rf", "bilstm", "distilbert"])
    ap.add_argument("--sample", type=int, default=None, help="use a random subset (smoke test)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)

    df = load_data(args.data, args.sample)
    train, val, test = split_data(df)
    print(f"URLs after dedup: {len(df):,} | train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"Phishing share (train): {train.y.mean():.3f}")

    runners = {"rf": ("Random Forest", run_rf), "bilstm": ("Char-BiLSTM", run_bilstm),
               "distilbert": ("DistilBERT", run_distilbert)}
    rows, probs, preds = [], {}, {}
    for key in args.models:
        name, fn = runners[key]
        prob, ms = fn(train, val, test)
        row, pred = compute_metrics(name, test.y.values, prob, ms)
        rows.append(row); probs[name] = prob; preds[name] = pred
        save_errors(name, test, pred)

    table = pd.DataFrame(rows)
    table.to_csv(f"{RESULTS_DIR}/results_table.csv", index=False)
    print(f"\nTest-set results (n = {len(test):,})")
    print(table.round(4).to_string(index=False))
    make_plots(test.y.values, probs, preds)
    print(f"\nSaved tables, plots and misclassified URLs to ./{RESULTS_DIR}/")


if __name__ == "__main__":
    main()
