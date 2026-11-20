"""
PRISM Experiment Suite — v3
============================
Changes in v3 (over v2):
  a. CV groups come from Louvain communities (networkx louvain_communities,
     seed 42); v2 used greedy modularity (Clauset-Newman-Moore).
  b. The ring signal in_ring_cluster is produced by the ring detector in
     generate_dataset.py, not written from the fraud label.
  c. New outputs: per-vector recall, economic impact, learned weights,
     weight sensitivity and the graph-detection figure.
  d. Tables print three decimals; captions no longer overclaim.

Key improvements of v2 over v1:
  1. One-Class SVM AUC corrected (negated score_samples → anomaly = high)
  2. Ring/community-level GroupKFold replaces random StratifiedKFold
     → prevents same fraud ring from leaking across train/val folds
  3. Temporal holdout added (Month 1-4 train, Month 5 val, Month 6 test)
  4. Pillar 2 ablation explicitly explained in output
  5. Absolute novelty claims removed from printed summaries
  6. Every metric audited and cross-checked

Experiments
-----------
1. Baseline comparison (6 models, GroupKFold CV)
2. Temporal holdout evaluation
3. Ablation study (GroupKFold, ring-aware)
4. Robustness sweep (fraud rate 5–30%)
5. Per-vector detection heatmap
6. Publication figures → figures/
7. LaTeX tables → docs/
"""

import argparse, sys, warnings
from pathlib import Path
from datetime import datetime

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
FIGURES = ROOT / "figures"
DOCS    = ROOT / "docs"
for d in [DATA, FIGURES, DOCS]:
    d.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
BLUE  = "#1A3A5C"; TEAL  = "#2E86AB"; ORANGE = "#E84855"
GREEN = "#3BB273"; GOLD  = "#F4A261"; GREY   = "#6C757D"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

FEATURE_COLS = [
    "phone_verified", "meets_threshold", "returned_in_window",
    "referee_account_age", "same_device", "same_ip_subnet",
    "both_discounts", "monthly_referral_count", "in_ring_cluster",
    "credit_requested_ghs", "public_code_exposure", "stacked_credits",
    "betweenness_centrality", "pagerank", "clustering_coeff",
]

PILLAR_FEATURES = {
    "Pillar 1 (Phone/Identity)":   ["phone_verified"],
    "Pillar 2 (Temporal Gate)":    ["returned_in_window"],
    "Pillar 3 (Threshold)":        ["meets_threshold"],
    "Pillar 4 (Caps/Rewards)":     ["credit_requested_ghs", "stacked_credits",
                                    "both_discounts", "monthly_referral_count"],
    "Pillar 5 (Graph/Ring)":       ["in_ring_cluster", "betweenness_centrality",
                                    "pagerank", "clustering_coeff",
                                    "public_code_exposure"],
}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def get_features(df):
    X = df[FEATURE_COLS].copy()
    X["referee_account_age"]    = X["referee_account_age"].clip(0, 365)
    X["credit_requested_ghs"]   = X["credit_requested_ghs"].clip(0, 200)
    X["monthly_referral_count"] = X["monthly_referral_count"].clip(0, 60)
    return X.fillna(0).astype(float)


def compute_metrics(y_true, y_pred, y_prob=None):
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    p   = tp/(tp+fp)   if tp+fp > 0 else 0
    r   = tp/(tp+fn)   if tp+fn > 0 else 0
    f1  = 2*p*r/(p+r)  if p+r   > 0 else 0
    fpr = fp/(fp+tn)   if fp+tn > 0 else 0
    auc = round(roc_auc_score(y_true, y_prob), 3) if y_prob is not None else None
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=round(p,3), Recall=round(r,3),
                F1=round(f1,3), FPR=round(fpr,3),
                AUC=auc if auc else "—")


# ══════════════════════════════════════════════════════════════════
# DATA LOADING WITH COMMUNITY GROUPS
# ══════════════════════════════════════════════════════════════════

def assign_groups(df, seed=42):
    """Louvain communities of the undirected referral graph, used as CV groups
    so that no community (and hence no ring) spans train and validation."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    G = nx.Graph()
    G.add_edges_from(zip(df["referrer_id"], df["referee_id"]))
    comms = louvain_communities(G, seed=seed)
    n2c = {n: i for i, c in enumerate(comms) for n in c}
    return df["referrer_id"].map(n2c).fillna(-1).astype(int)


def load_dataset(verbose=True):
    grouped_path = DATA / "prism_dataset_with_groups.csv"
    base_path    = DATA / "prism_synthetic_dataset.csv"

    if grouped_path.exists():
        df = pd.read_csv(grouped_path)
        if verbose:
            print(f"[1] Loaded grouped dataset: {len(df):,} records")
    elif base_path.exists():
        if verbose:
            print("[1] Grouped dataset not found — assigning community groups...")
        df = pd.read_csv(base_path)
        df["group_id"] = assign_groups(df)
        df.to_csv(grouped_path, index=False)
    else:
        raise FileNotFoundError("No dataset found. Run generate_dataset.py first.")

    # Parse order_date for temporal split
    df["order_date"] = pd.to_datetime(df["order_date"])
    if verbose:
        print(f"    Groups: {df['group_id'].nunique()}  "
              f"Fraud: {df['fraud_label'].mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — BASELINE COMPARISON (GroupKFold)
# ══════════════════════════════════════════════════════════════════

def run_baselines(df, seed=42, verbose=True):
    """
    Ring-aware GroupKFold cross-validation.
    Groups = Louvain communities so no fraud ring spans train/val.
    """
    if verbose:
        print("\n[2] Baseline comparison (GroupKFold, ring-aware)...")

    y      = df["fraud_label"].values
    groups = df["group_id"].values
    X      = get_features(df)
    Xs     = StandardScaler().fit_transform(X)
    cv     = GroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups))

    results = {}

    # ── Rule-based PRISM ──────────────────────────────────────────
    rule_pred  = (df["prism_decision"] == "hold_review").astype(int).values
    rule_score = -df["prism_score"].values
    rs_max = rule_score.max() or 1
    results["PRISM (Rule-based)"] = compute_metrics(y, rule_pred,
                                                    rule_score / rs_max)

    # ── PRISM-LR (GroupKFold) ─────────────────────────────────────
    if verbose: print("    PRISM-LR...")
    lr = LogisticRegression(C=1.0, class_weight="balanced",
                            max_iter=1000, random_state=seed, solver="lbfgs")
    lr_probs = np.zeros(len(y))
    for tr, va in splits:
        sc = StandardScaler().fit(X.iloc[tr])
        lr.fit(sc.transform(X.iloc[tr]), y[tr])
        lr_probs[va] = lr.predict_proba(sc.transform(X.iloc[va]))[:, 1]
    lr_pred = (lr_probs >= 0.5).astype(int)
    results["PRISM-LR (Learned)"] = compute_metrics(y, lr_pred, lr_probs)
    # refit on full data for coefficient table
    sc_full = StandardScaler().fit(X)
    lr_full = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000, random_state=seed)
    lr_full.fit(sc_full.transform(X), y)

    # ── Random Forest ─────────────────────────────────────────────
    if verbose: print("    Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf_probs = np.zeros(len(y))
    for tr, va in splits:
        rf.fit(X.iloc[tr], y[tr])
        rf_probs[va] = rf.predict_proba(X.iloc[va])[:, 1]
    results["Random Forest"] = compute_metrics(y, (rf_probs>=0.5).astype(int),
                                               rf_probs)

    # ── XGBoost ───────────────────────────────────────────────────
    if verbose: print("    XGBoost...")
    sp = (y==0).sum() / (y==1).sum()
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                scale_pos_weight=sp, random_state=seed,
                                eval_metric="logloss", verbosity=0)
    xgb_probs = np.zeros(len(y))
    for tr, va in splits:
        xgb_m.fit(X.iloc[tr], y[tr])
        xgb_probs[va] = xgb_m.predict_proba(X.iloc[va])[:, 1]
    results["XGBoost"] = compute_metrics(y, (xgb_probs>=0.5).astype(int),
                                         xgb_probs)

    # ── Isolation Forest ──────────────────────────────────────────
    if verbose: print("    Isolation Forest...")
    iso_probs = np.zeros(len(y))
    iso_pred  = np.zeros(len(y), dtype=int)
    for tr, va in splits:
        iso = IsolationForest(contamination=0.20, random_state=seed, n_jobs=-1)
        iso.fit(Xs[tr])
        sc_ = -iso.score_samples(Xs[va])  # high = anomaly
        iso_probs[va] = sc_
        iso_pred[va]  = (iso.predict(Xs[va]) == -1).astype(int)
    iso_probs = iso_probs / (iso_probs.max() or 1)
    results["Isolation Forest"] = compute_metrics(y, iso_pred, iso_probs)

    # ── One-Class SVM ─────────────────────────────────────────────
    # NOTE: score_samples returns higher values for INLIERS (legitimate).
    # We NEGATE the score so that high values = anomaly = fraud.
    # AUC is computed on the negated score to align positive class = fraud.
    if verbose: print("    One-Class SVM...")
    ocsvm_probs = np.zeros(len(y))
    ocsvm_pred  = np.zeros(len(y), dtype=int)
    for tr, va in splits:
        # Train only on legitimate events in the training fold
        legit_tr = tr[y[tr] == 0]
        ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
        ocsvm.fit(Xs[legit_tr])
        # Negate: low inlier score → high anomaly score
        ocsvm_probs[va] = -ocsvm.score_samples(Xs[va])
        ocsvm_pred[va]  = (ocsvm.predict(Xs[va]) == -1).astype(int)
    # Global min-max after all folds so AUC is computed correctly
    _mn = ocsvm_probs.min(); _mx = ocsvm_probs.max()
    ocsvm_probs = (ocsvm_probs - _mn) / (_mx - _mn + 1e-9)
    results["One-Class SVM"] = compute_metrics(y, ocsvm_pred, ocsvm_probs)

    if verbose:
        print(f"\n  {'Method':<25} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'FPR':>6} {'AUC':>7}")
        print(f"  {'-'*57}")
        for method, m in results.items():
            mk = "* " if method.startswith("PRISM") else "  "
            print(f"  {mk}{method:<23} {m['Precision']:>6} {m['Recall']:>6} "
                  f"{m['F1']:>6} {m['FPR']:>6} {m['AUC']:>7}")

    return results, lr_probs, rf_probs, xgb_probs, rule_score, y, X, Xs


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — TEMPORAL HOLDOUT
# ══════════════════════════════════════════════════════════════════

def run_temporal_holdout(df, seed=42, verbose=True):
    """
    Temporal split: train=months 1–4, val=month 5, test=month 6.
    Simulates real deployment where the model never sees future fraud patterns.
    """
    if verbose:
        print("\n[3] Temporal holdout evaluation...")

    start = df["order_date"].min()
    cutoff_train = start + pd.Timedelta(days=120)  # month 4
    cutoff_val   = start + pd.Timedelta(days=150)  # month 5

    train_mask = df["order_date"] <= cutoff_train
    val_mask   = (df["order_date"] > cutoff_train) & (df["order_date"] <= cutoff_val)
    test_mask  = df["order_date"] > cutoff_val

    if verbose:
        print(f"    Train: {train_mask.sum():,}  Val: {val_mask.sum():,}  "
              f"Test: {test_mask.sum():,}")

    X = get_features(df)
    y = df["fraud_label"].values
    sc = StandardScaler().fit(X[train_mask])
    X_tr = sc.transform(X[train_mask])
    X_te = sc.transform(X[test_mask])
    y_tr = y[train_mask]
    y_te = y[test_mask]

    results_temporal = {}

    # Rule-based (no training needed)
    rule_pred = (df["prism_decision"] == "hold_review").astype(int).values
    rule_score = -df["prism_score"].values
    results_temporal["PRISM (Rule-based)"] = compute_metrics(
        y_te, rule_pred[test_mask],
        (rule_score / (rule_score.max() or 1))[test_mask])

    # PRISM-LR
    lr = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=1000, random_state=seed)
    lr.fit(X_tr, y_tr)
    lr_p = lr.predict_proba(X_te)[:, 1]
    results_temporal["PRISM-LR"] = compute_metrics(y_te, (lr_p>=0.5).astype(int), lr_p)

    # XGBoost
    sp = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=sp, random_state=seed,
                             eval_metric="logloss", verbosity=0)
    xg.fit(X_tr, y_tr)
    xg_p = xg.predict_proba(X_te)[:, 1]
    results_temporal["XGBoost"] = compute_metrics(y_te, (xg_p>=0.5).astype(int), xg_p)

    if verbose:
        print(f"\n  {'Method':<25} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7}")
        print(f"  {'-'*50}")
        for method, m in results_temporal.items():
            print(f"  {method:<25} {m['Precision']:>6} {m['Recall']:>6} "
                  f"{m['F1']:>6} {m['AUC']:>7}")
    return results_temporal


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — ABLATION (GroupKFold, ring-aware)
# ══════════════════════════════════════════════════════════════════

def run_ablation(df, y, seed=42, verbose=True):
    if verbose:
        print("\n[4] Ablation study (GroupKFold, ring-aware)...")

    groups = df["group_id"].values
    cv = GroupKFold(n_splits=5)
    ablation = {}

    configs = {"Full PRISM": []}
    for name, cols in PILLAR_FEATURES.items():
        configs[f"w/o {name}"] = cols

    full_f1 = None
    for label, drop_cols in configs.items():
        X_ab = get_features(df).drop(columns=drop_cols, errors="ignore")
        lr_ab = LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=1000, random_state=seed)
        probs = np.zeros(len(y))
        for tr, va in cv.split(X_ab, y, groups):
            sc = StandardScaler().fit(X_ab.iloc[tr])
            lr_ab.fit(sc.transform(X_ab.iloc[tr]), y[tr])
            probs[va] = lr_ab.predict_proba(sc.transform(X_ab.iloc[va]))[:, 1]
        m = compute_metrics(y, (probs>=0.5).astype(int), probs)
        ablation[label] = m
        if label == "Full PRISM":
            full_f1 = m["F1"]

        delta = f"{m['F1']-full_f1:+.3f}" if full_f1 and label != "Full PRISM" else "—"
        note  = ""
        if label == "w/o Pillar 2 (Temporal Gate)":
            note = "  [Pillar 2 is also enforced as a hard gate before scoring]"
        if verbose:
            print(f"    {label:<35} F1={m['F1']:.3f}  ΔF1={delta}{note}")

    return ablation


# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — ROBUSTNESS SWEEP
# ══════════════════════════════════════════════════════════════════

def run_robustness(seed=42, verbose=True):
    if verbose:
        print("\n[5] Robustness sweep (5%–30% fraud rate)...")

    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import generate_dataset

    fraud_rates = [0.05, 0.10, 0.20, 0.30]
    robustness  = {}

    for fr in fraud_rates:
        if verbose:
            print(f"    fraud_rate={fr:.0%}...", end=" ", flush=True)
        df_r, _, _ = generate_dataset(fraud_rate=fr, total=10_000, verbose=False)
        df_r["order_date"] = pd.to_datetime(df_r["order_date"])

        # Assign communities for group splitting
        df_r["group_id"] = assign_groups(df_r, seed)

        y_r      = df_r["fraud_label"].values
        groups_r = df_r["group_id"].values
        X_r      = get_features(df_r)
        cv_r     = GroupKFold(n_splits=5)
        splits_r = list(cv_r.split(X_r, y_r, groups_r))

        # Rule-based
        rp = (df_r["prism_decision"] == "hold_review").astype(int).values
        rs = -df_r["prism_score"].values
        rule_m = compute_metrics(y_r, rp, rs / (rs.max() or 1))

        # PRISM-LR
        lr_r   = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=1000, random_state=seed)
        lr_pr  = np.zeros(len(y_r))
        for tr, va in splits_r:
            sc = StandardScaler().fit(X_r.iloc[tr])
            lr_r.fit(sc.transform(X_r.iloc[tr]), y_r[tr])
            lr_pr[va] = lr_r.predict_proba(sc.transform(X_r.iloc[va]))[:, 1]
        lr_m = compute_metrics(y_r, (lr_pr>=0.5).astype(int), lr_pr)

        # XGBoost
        sp_r   = (y_r==0).sum() / max((y_r==1).sum(), 1)
        xg_r   = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                     scale_pos_weight=sp_r, random_state=seed,
                                     eval_metric="logloss", verbosity=0)
        xg_pr  = np.zeros(len(y_r))
        for tr, va in splits_r:
            xg_r.fit(X_r.iloc[tr], y_r[tr])
            xg_pr[va] = xg_r.predict_proba(X_r.iloc[va])[:, 1]
        xg_m = compute_metrics(y_r, (xg_pr>=0.5).astype(int), xg_pr)

        robustness[fr] = {"Rule-based": rule_m, "LR": lr_m, "XGBoost": xg_m}
        if verbose:
            print(f"Rule F1={rule_m['F1']:.3f}  "
                  f"LR F1={lr_m['F1']:.3f}  XGB F1={xg_m['F1']:.3f}")

    return robustness


# ══════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════

def fig_baseline(results):
    methods = list(results.keys())
    metric_names = ["Precision","Recall","F1","FPR"]
    colours = [TEAL, BLUE, GREEN, ORANGE]
    x = np.arange(len(methods)); w = 0.20

    fig, ax = plt.subplots(figsize=(13,6))
    for i, (metric, col) in enumerate(zip(metric_names, colours)):
        vals = [results[m][metric] for m in methods]
        bars = ax.bar(x+(i-1.5)*w, vals, w, label=metric,
                      color=col, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=18, ha="right", fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("PRISM vs Baseline Methods — GroupKFold CV (N=10,000)",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(FIGURES/"fig_baseline_comparison.png"); plt.close()
    print("  ✓ fig_baseline_comparison.png")


def fig_roc(y, rule_score, lr_probs, rf_probs, xgb_probs):
    roc_data = {
        "PRISM (Rule-based)": rule_score / (rule_score.max() or 1),
        "PRISM-LR":           lr_probs,
        "Random Forest":      rf_probs,
        "XGBoost":            xgb_probs,
    }
    colours = [ORANGE, TEAL, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7,6))
    for (name, probs), col in zip(roc_data.items(), colours):
        fpr_, tpr_, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr_, tpr_, color=col, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--",linewidth=1,alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves — GroupKFold CV", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES/"fig_roc_curves.png"); plt.close()
    print("  ✓ fig_roc_curves.png")


def fig_ablation(ablation, full_f1):
    labels = list(ablation.keys())
    f1s    = [ablation[l]["F1"] for l in labels]
    drops  = [full_f1 - f for f in f1s]

    fig, axes = plt.subplots(1,2,figsize=(14,6))
    cols_ab = [TEAL]+[ORANGE if d>0.005 else (GOLD if d>0 else GREEN)
                      for d in drops[1:]]

    ax = axes[0]
    bars = ax.barh(labels[::-1], f1s[::-1],
                   color=cols_ab[::-1], edgecolor="white", alpha=0.88, height=0.6)
    for bar, v in zip(bars, f1s[::-1]):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10, fontweight="bold", color=BLUE)
    ax.axvline(full_f1, color=TEAL, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("F1-Score", fontweight="bold")
    ax.set_title("(a) Ablation: F1-Score", fontweight="bold")
    ax.set_xlim(0, 1.1); ax.grid(axis="x", alpha=0.2)

    ax2 = axes[1]
    drop_labels = labels[1:]
    drop_vals   = [full_f1 - ablation[l]["F1"] for l in drop_labels]
    cols_d = [ORANGE if d>0.005 else (GOLD if d>0 else GREEN) for d in drop_vals]
    bars2 = ax2.barh(drop_labels[::-1], drop_vals[::-1],
                     color=cols_d[::-1], edgecolor="white", alpha=0.88, height=0.55)
    for bar, v, lbl in zip(bars2, drop_vals[::-1], drop_labels[::-1]):
        sign = "−" if v > 0 else "+"
        ax2.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                 f"{sign}{abs(v):.3f}", va="center", fontsize=10,
                 fontweight="bold", color=BLUE)
    ax2.axvline(0, color="grey", linewidth=1, alpha=0.5)
    ax2.set_xlabel("ΔF1 (negative = performance drop)", fontweight="bold")
    ax2.set_title("(b) Ablation: ΔF1 from Full PRISM\n"
                  "(Pillar 2 is enforced as a hard gate before scoring)",
                  fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES/"fig_ablation.png"); plt.close()
    print("  ✓ fig_ablation.png")


def fig_robustness(robustness):
    fraud_rates = list(robustness.keys())
    fr_labels   = [f"{int(fr*100)}%" for fr in fraud_rates]
    methods_r   = ["Rule-based","LR","XGBoost"]
    colours_r   = [ORANGE, TEAL, BLUE]

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax_i, metric in enumerate(["F1","Recall","Precision"]):
        ax = axes[ax_i]
        for method, col in zip(methods_r, colours_r):
            vals = [robustness[fr][method][metric] for fr in fraud_rates]
            ax.plot(fr_labels, vals, marker="o", color=col,
                    linewidth=2.5, markersize=8, label=method)
            for xi, v in enumerate(vals):
                ax.annotate(f"{v:.2f}", (fr_labels[xi], v),
                            textcoords="offset points", xytext=(0,8),
                            ha="center", fontsize=9, color=col)
        ax.set_xlabel("Fraud Rate", fontweight="bold")
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_title(f"({chr(97+ax_i)}) {metric} vs Fraud Rate",
                     fontweight="bold")
        ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
    fig.suptitle("PRISM Robustness Across Fraud Rates (GroupKFold CV)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES/"fig_robustness.png"); plt.close()
    print("  ✓ fig_robustness.png")


def fig_temporal(results_temporal):
    methods = list(results_temporal.keys())
    metrics_t = ["Precision","Recall","F1"]
    colours_t = [TEAL, GREEN, ORANGE]
    x = np.arange(len(methods)); w = 0.25

    fig, ax = plt.subplots(figsize=(9,5))
    for i, (metric, col) in enumerate(zip(metrics_t, colours_t)):
        vals = [results_temporal[m][metric] for m in methods]
        bars = ax.bar(x+(i-1)*w, vals, w, label=metric,
                      color=col, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0,1.18); ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Temporal Holdout: Train months 1–4, Test month 6",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES/"fig_temporal_holdout.png"); plt.close()
    print("  ✓ fig_temporal_holdout.png")


# ══════════════════════════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════════════════════════

def _f(v):
    return f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v)


def write_latex_tables(results, ablation, robustness, results_temporal):
    lines = ["% PRISM Experiment Tables (auto-generated by run_experiments.py v3)\n",
             "% CV method: GroupKFold, groups = Louvain communities (seed 42)\n",
             "% One-Class SVM AUC: negated score_samples (anomaly = high score)\n\n"]

    # Baseline
    lines += [
        r"\begin{table*}[ht]", r"\centering",
        r"\caption{Detection Performance: PRISM vs Baseline Methods "
        r"(5-fold GroupKFold CV, $N=10{,}000$, fraud rate 20\%). "
        r"GroupKFold ensures no fraud ring spans train and validation folds. "
        r"One-Class SVM AUC computed on negated \texttt{score\_samples} "
        r"so that high score $\equiv$ anomaly.}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Precision} & \textbf{Recall} & "
        r"\textbf{F1} & \textbf{FPR} & \textbf{AUC} \\",
        r"\midrule",
    ]
    for method, m in results.items():
        b = r"\textbf{" if method.startswith("PRISM") else ""
        e = "}" if method.startswith("PRISM") else ""
        lines.append(f"{b}{method}{e} & {b}{_f(m['Precision'])}{e} & "
                     f"{b}{_f(m['Recall'])}{e} & {b}{_f(m['F1'])}{e} & "
                     f"{b}{_f(m['FPR'])}{e} & {b}{_f(m['AUC'])}{e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    # Temporal holdout
    lines += [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Temporal holdout: models trained on events up to day 120 "
        r"(months 1--4) and tested on events after day 150 (month 6). "
        r"Rule-based PRISM needs no training. The drop relative to "
        r"Table~\ref{tab:baseline} indicates sensitivity to temporal drift.}",
        r"\label{tab:temporal}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Precision} & \textbf{Recall} & "
        r"\textbf{F1} & \textbf{AUC} \\",
        r"\midrule",
    ]
    for method, m in results_temporal.items():
        b = r"\textbf{" if "PRISM" in method else ""
        e = "}" if "PRISM" in method else ""
        lines.append(f"{b}{method}{e} & {b}{_f(m['Precision'])}{e} & "
                     f"{b}{_f(m['Recall'])}{e} & {b}{_f(m['F1'])}{e} & "
                     f"{b}{_f(m['AUC'])}{e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Ablation
    full_f1 = ablation["Full PRISM"]["F1"]
    lines += [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Ablation study: PRISM-LR with each pillar's features "
        r"removed (GroupKFold CV). Pillar~2 is enforced as a hard gate "
        r"before scoring, so removing its feature from the classifier "
        r"does not remove its operational effect.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{P} & \textbf{R} & "
        r"\textbf{F1} & \textbf{$\Delta$F1} \\",
        r"\midrule",
    ]
    for label, m in ablation.items():
        delta = f"{m['F1']-full_f1:+.3f}" if label != "Full PRISM" else "---"
        b = r"\textbf{" if label == "Full PRISM" else ""
        e = "}" if label == "Full PRISM" else ""
        lines.append(f"{b}{label}{e} & "
                     f"{b}{_f(m['Precision'])}{e} & {b}{_f(m['Recall'])}{e} & "
                     f"{b}{_f(m['F1'])}{e} & {b}{delta}{e} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Robustness
    fr_list = list(robustness.keys())
    lines += [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Robustness: F1-score across fraud prevalence rates "
        r"(GroupKFold CV, $N=10{,}000$ per rate). Performance of both "
        r"PRISM variants falls sharply at low prevalence.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{5\%} & \textbf{10\%} & "
        r"\textbf{20\%} & \textbf{30\%} \\",
        r"\midrule",
    ]
    for method_key, label in [("Rule-based","PRISM Rule-based"),
                               ("LR","PRISM-LR"),
                               ("XGBoost","XGBoost")]:
        vals = " & ".join(f"{robustness[fr][method_key]['F1']:.3f}"
                          for fr in fr_list)
        b = r"\textbf{" if "PRISM" in label else ""
        e = "}" if "PRISM" in label else ""
        lines.append(f"{b}{label}{e} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    lines += [""] + EXTRA_TABLES
    path = DOCS / "prism_experiment_tables.tex"
    path.write_text("\n".join(lines))
    print(f"  ✓ prism_experiment_tables.tex")


# ══════════════════════════════════════════════════════════════════
# EXPERIMENTS 5-8 (v3) — PER-VECTOR, ECONOMIC, WEIGHTS, SENSITIVITY
# ══════════════════════════════════════════════════════════════════

EXTRA_TABLES = []          # LaTeX tables appended by the v3 experiments
REVIEW_COST_GHS = 5        # manual review cost per held legitimate credit

VECTOR_NAMES = {
    "E1_self_referral": "E1 Self-referral", "E2_fake_farming": "E2 Fake farming",
    "E3_micro_order": "E3 Micro-order", "E4_threshold_return": "E4 Threshold+return",
    "E5_credit_stacking": "E5 Credit stacking", "E6_discount_compound": "E6 Discount compounding",
    "E7_referral_ring": "E7 Referral ring", "E8_code_exposure": "E8 Code exposure",
}


def run_per_vector(df, preds, verbose=True):
    """Recall per exploitation vector for each method (binary predictions)."""
    if verbose:
        print("\n[6] Per-vector recall...")
    fraud = df["fraud_label"].values == 1
    out = {}
    for vec in VECTOR_NAMES:
        mask = (df["fraud_type"] == vec).values
        out[vec] = {m: float(p[mask].mean()) for m, p in preds.items()}
        out[vec]["n"] = int(mask.sum())
        if verbose:
            print(f"    {vec:<24} " + "  ".join(f"{m}={out[vec][m]:.3f}" for m in preds))
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Per-vector recall (GroupKFold CV for learned models).}",
             r"\label{tab:pervector}",
             r"\begin{tabular}{@{}l" + "r" * (len(preds) + 1) + r"@{}}", r"\toprule",
             r"\textbf{Vector} & \textbf{$n$} & " +
             " & ".join(rf"\textbf{{{m}}}" for m in preds) + r" \\", r"\midrule"]
    for vec, row in out.items():
        lines.append(f"{VECTOR_NAMES[vec]} & {row['n']} & " +
                     " & ".join(f"{row[m]:.3f}" for m in preds) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    EXTRA_TABLES.extend(lines)

    mat = np.array([[out[v][m] for m in preds] for v in VECTOR_NAMES])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(preds))); ax.set_xticklabels(list(preds), fontsize=10)
    ax.set_yticks(range(len(VECTOR_NAMES)))
    ax.set_yticklabels(list(VECTOR_NAMES.values()), fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if mat[i, j] > 0.6 else BLUE)
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Per-vector recall", fontweight="bold")
    plt.tight_layout(); plt.savefig(FIGURES / "fig_per_vector_heatmap.png"); plt.close()
    print("  ✓ fig_per_vector_heatmap.png")
    return out


def run_economic(df, verbose=True):
    """Credits protected and leaked by rule-based PRISM over the 180-day window."""
    y = df["fraud_label"].values == 1
    hold = (df["prism_decision"] == "hold_review").values
    credit = df["credit_requested_ghs"].values
    exposure = credit[y].sum(); protected = credit[y & hold].sum()
    leaked = exposure - protected
    n_fp = int((hold & ~y).sum()); review = n_fp * REVIEW_COST_GHS
    net = protected - review
    eco = dict(exposure=exposure, protected=protected, leaked=leaked, n_fp=n_fp,
               review=review, net=net, reduction=100 * protected / exposure,
               residual=leaked + review, ratio=net / review if review else float("nan"))
    if verbose:
        print("\n[7] Economic impact (rule-based PRISM)...")
        for k, v in eco.items():
            print(f"    {k:<10} {v:,.1f}")
    g = lambda v: f"{v:,.0f}".replace(",", "{,}")
    EXTRA_TABLES.extend([
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Economic impact of rule-based PRISM over the 180-day "
        r"simulation window, against an uncontrolled baseline.}",
        r"\label{tab:economic}", r"\begin{tabular}{@{}lr@{}}", r"\toprule",
        r"\textbf{Economic Metric} & \textbf{Value (GHS)} \\", r"\midrule",
        rf"Total fraud exposure (no controls) & {g(exposure)} \\",
        rf"Credits protected by PRISM (TP) & {g(protected)} \\",
        rf"Credits leaked through PRISM (FN) & {g(leaked)} \\",
        rf"Manual review cost ({n_fp} $\times$ GHS {REVIEW_COST_GHS}) & {g(review)} \\",
        rf"Net saving (protected $-$ review cost) & {g(net)} \\",
        rf"Fraud exposure reduction & {eco['reduction']:.1f}\% \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return eco


RULE_FEATURE_MAP = [  # (rule weight key, rule signal, LR feature)
    ("account_age",    r"Account age $< 7$ days",       "referee_account_age"),
    ("same_device",    "Same device fingerprint",       "same_device"),
    ("both_discounts", "Both discounts applied",        "both_discounts"),
    ("monthly_cap",    r"Monthly count $> 15$",         "monthly_referral_count"),
    ("ring_cluster",   "In ring cluster",               "in_ring_cluster"),
    ("credit_cap",     r"Credit requested $>$ cap",     "credit_requested_ghs"),
    ("code_exposure",  "Code velocity spike",           "public_code_exposure"),
    ("compound",       "Compound (device + IP)",        "same_ip_subnet"),
]


def run_learned_weights(df, seed=42, verbose=True):
    """Standardised LR coefficients (full-data refit) next to rule weights.
    Positive coefficient = raises fraud probability."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import RULE_WEIGHTS
    X = get_features(df); y = df["fraud_label"].values
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                            random_state=seed).fit(sc.transform(X), y)
    coef = dict(zip(FEATURE_COLS, lr.coef_[0]))
    if verbose:
        print("\n[8] Learned weights (standardised LR coefficients)...")
        for c, w in sorted(coef.items(), key=lambda z: -abs(z[1])):
            print(f"    {c:<26} {w:+.3f}")
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Rule-based weights and PRISM-LR coefficients "
             r"(standardised features, fitted on the full dataset). A positive "
             r"coefficient raises the predicted fraud probability; the rule "
             r"weights are penalties, so a negative rule weight corresponds to "
             r"a positive coefficient. Account age enters PRISM-LR as a "
             r"continuous feature, so its coefficient is negative (older "
             r"accounts are less likely to be fraudulent).}",
             r"\label{tab:learned_weights}", r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"\textbf{Signal} & \textbf{LR feature} & \textbf{Rule $w_j$} & "
             r"\textbf{LR coef.} \\", r"\midrule"]
    used = set()
    for key, label, feat in RULE_FEATURE_MAP:
        used.add(feat)
        lines.append(rf"{label} & \texttt{{{feat.replace('_', chr(92) + '_')}}} & "
                     rf"${RULE_WEIGHTS[key]}$ & ${coef[feat]:+.3f}$ \\")
    lines.append(r"\midrule")
    for feat in FEATURE_COLS:
        if feat not in used:
            lines.append(rf"(not in rule score) & \texttt{{{feat.replace('_', chr(92) + '_')}}} & "
                         rf"--- & ${coef[feat]:+.3f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    EXTRA_TABLES.extend(lines)
    return coef


def run_sensitivity(df, verbose=True):
    """Perturb each rule weight by ±50%. The release/hold decision is
    'hold if any signal fires', so F1 and recall cannot change; only the
    ordering of held events (AUC of the score) can."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import RULE_WEIGHTS, prism_score, prism_decision
    y = df["fraud_label"].values
    base_auc = roc_auc_score(y, -df["prism_score"].values)
    rows = []
    for key in RULE_WEIGHTS:
        res = {}
        for f in (0.5, 1.5):
            w = dict(RULE_WEIGHTS); w[key] = RULE_WEIGHTS[key] * f
            sc = df.apply(lambda r: prism_score(r, w), axis=1)
            pred = (sc.apply(prism_decision) == "hold_review").astype(int).values
            res[f] = (roc_auc_score(y, -sc.values), compute_metrics(y, pred)["F1"])
        rows.append((key, res[0.5], res[1.5]))
    f1_base = compute_metrics(y, (df["prism_decision"] == "hold_review").astype(int).values)["F1"]
    max_df1 = max(abs(r[i][1] - f1_base) for r in rows for i in (1, 2))
    if verbose:
        print(f"\n[9] Weight sensitivity (base AUC={base_auc:.3f}, max |ΔF1|={max_df1:.3f})")
        for k, lo, hi in rows:
            print(f"    {k:<15} AUC -50%={lo[0]:.3f}  +50%={hi[0]:.3f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(rows, key=lambda r: abs(r[2][0] - r[1][0]))
    for i, (k, lo, hi) in enumerate(order):
        ax.barh(i, lo[0] - base_auc, color=TEAL, alpha=0.85, label="−50%" if i == 0 else None)
        ax.barh(i, hi[0] - base_auc, color=ORANGE, alpha=0.85, label="+50%" if i == 0 else None)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([r[0] for r in order])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel(f"ΔAUC of rule-based score (base AUC = {base_auc:.3f})", fontweight="bold")
    ax.set_title("Rule-weight sensitivity (F1 unchanged by construction)", fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.savefig(FIGURES / "fig6_sensitivity.png"); plt.close()
    print("  ✓ fig6_sensitivity.png")
    return dict(base_auc=base_auc, rows=rows, max_df1=max_df1, f1_base=f1_base)


def fig_graph(df):
    """(a) Louvain communities of the full graph; (b) communities of the
    reciprocal subgraph used by the ring detector."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    G = nx.DiGraph(); G.add_edges_from(zip(df["referrer_id"], df["referee_id"]))
    e7 = set(df.loc[df["fraud_type"] == "E7_referral_ring", "referrer_id"])

    def stats(comms):
        out = []
        for c in comms:
            s = G.subgraph(c); ne = s.number_of_edges()
            nr = sum(1 for u, v in s.edges() if s.has_edge(v, u))
            out.append((len(c), nr / ne if ne else 0, len(set(c) & e7) / len(c)))
        return np.array(out)

    full = stats(louvain_communities(G.to_undirected(), seed=42))
    GR = nx.Graph([(u, v) for u, v in G.edges() if G.has_edge(v, u)])
    rec = stats(louvain_communities(GR, seed=42)) if GR.number_of_edges() else np.zeros((0, 3))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, st, title in [(axes[0], full, "(a) Louvain on full graph"),
                          (axes[1], rec, "(b) Louvain on reciprocal subgraph")]:
        ring = st[:, 2] > 0.5 if len(st) else np.array([], bool)
        ax.scatter(st[~ring, 0], st[~ring, 1], c=TEAL, s=40, label="Other communities")
        ax.scatter(st[ring, 0], st[ring, 1], c=ORANGE, marker="*", s=160,
                   label="Majority E7 ring members")
        ax.axhline(0.6, ls="--", color="grey"); ax.axvline(5, ls=":", color="grey")
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Community size |C| (log)", fontweight="bold")
        ax.set_ylabel(r"Reciprocal edge density $\rho$", fontweight="bold")
        ax.set_title(title, fontweight="bold"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(FIGURES / "fig5_graph_detection.png"); plt.close()
    print("  ✓ fig5_graph_detection.png")
    n_full_ring = int(((full[:, 0] >= 5) & (full[:, 1] >= 0.6)).sum())
    return dict(full_communities=len(full), full_flagged=n_full_ring,
                rec_communities=len(rec))


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PRISM Experiment Suite v2")
    parser.add_argument("--quick", action="store_true",
                        help="Skip robustness sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed; np.random.seed(seed)

    print("=" * 60)
    print("  PRISM Experiment Suite v3")
    print("  GroupKFold CV | Temporal Holdout | AUC-corrected")
    print("=" * 60)

    df = load_dataset()
    y  = df["fraud_label"].values

    results, lr_probs, rf_probs, xgb_probs, rule_score, y, X, Xs = \
        run_baselines(df, seed)

    results_temporal = run_temporal_holdout(df, seed)

    ablation = run_ablation(df, y, seed)
    full_f1  = ablation["Full PRISM"]["F1"]

    if not args.quick:
        robustness = run_robustness(seed)
    else:
        print("\n[5] Robustness skipped (--quick)")
        robustness = None

    lr_pred  = (lr_probs >= 0.5).astype(int)
    preds = {"Rule-based": (df["prism_decision"] == "hold_review").astype(int).values,
             "PRISM-LR": lr_pred,
             "RF": (rf_probs >= 0.5).astype(int),
             "XGBoost": (xgb_probs >= 0.5).astype(int)}
    per_vector = run_per_vector(df, preds)
    eco   = run_economic(df)
    coef  = run_learned_weights(df, seed)
    sens  = run_sensitivity(df)
    ring_e7 = df.loc[df["fraud_type"] == "E7_referral_ring", "in_ring_cluster"].mean()
    ring_fp = int(df.loc[df["fraud_label"] == 0, "in_ring_cluster"].sum())
    print(f"\n    Ring detector: E7 recall={ring_e7:.3f}  legit events flagged={ring_fp}")

    print("\n[10] Generating figures...")
    gstats = fig_graph(df)
    print(f"    Graph: full-graph communities={gstats['full_communities']} "
          f"(meeting ring criterion: {gstats['full_flagged']}); "
          f"reciprocal-subgraph communities={gstats['rec_communities']}")
    fig_baseline(results)
    fig_roc(y, rule_score, lr_probs, rf_probs, xgb_probs)
    fig_ablation(ablation, full_f1)
    fig_temporal(results_temporal)
    if robustness:
        fig_robustness(robustness)

    print("\n[11] Writing LaTeX tables...")
    if robustness:
        write_latex_tables(results, ablation, robustness, results_temporal)

    print(f"\n{'='*60}")
    print("  KEY RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  PRISM Rule-based: F1={results['PRISM (Rule-based)']['F1']}  "
          f"AUC={results['PRISM (Rule-based)']['AUC']}")
    print(f"  PRISM-LR:         F1={results['PRISM-LR (Learned)']['F1']}  "
          f"AUC={results['PRISM-LR (Learned)']['AUC']}")
    print(f"  Random Forest:    F1={results['Random Forest']['F1']}  "
          f"AUC={results['Random Forest']['AUC']}")
    print(f"  XGBoost:          F1={results['XGBoost']['F1']}  "
          f"AUC={results['XGBoost']['AUC']}")
    print(f"  One-Class SVM:    F1={results['One-Class SVM']['F1']}  "
          f"AUC={results['One-Class SVM']['AUC']}")
    ldr = max((l for l in ablation if l!="Full PRISM"),
              key=lambda l: full_f1-ablation[l]["F1"])
    print(f"\n  Ablation: largest drop from '{ldr}'  "
          f"ΔF1={full_f1-ablation[ldr]['F1']:+.3f}")
    print(f"\n  Temporal holdout PRISM-LR: "
          f"F1={results_temporal['PRISM-LR']['F1']}  "
          f"AUC={results_temporal['PRISM-LR']['AUC']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
