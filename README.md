# PRISM: A Fraud-Resistant Design Framework for Mobile-First Referral Programmes in Developing Markets

**Repository for the paper submitted to Expert Systems with Applications (Elsevier)**

Authors: Tonon Abdul-Shafiu Mahmud, Gaddafi Abdul-Salaam  
Affiliation: Department of Computer Science, KNUST, Kumasi, Ghana  
Contact: tasmahmud@knust.edu.gh

---

## Repository Contents

```
PRISM/
├── data/
│   ├── prism_synthetic_dataset.csv        # N=10,000 synthetic referral events (seed 42)
│   ├── prism_dataset_with_groups.csv      # same, with community group_id for GroupKFold
│   └── prism_community_stats.csv          # community detection results
├── src/
│   ├── generate_dataset.py                # synthetic dataset generator
│   ├── run_experiments.py                 # experiment suite (v3)
│   └── prism_scorer.py                    # standalone PRISM scorer (rule-based / PRISM-LR)
├── figures/                               # publication figures (300 DPI PNG)
├── docs/                                  # LaTeX tables and console log of the reported run
├── requirements.txt
├── README.md
└── LICENSE                                # CC BY 4.0
```

---

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

Versions are pinned to the environment used for the reported results (Python 3.13.18).
Other library versions can shift tree-ensemble results by about 0.02 F1.

### Step 1: Generate the synthetic dataset

```bash
python src/generate_dataset.py --output data/prism_synthetic_dataset.csv \
    --communities-output data/prism_community_stats.csv
```

This generates `data/prism_synthetic_dataset.csv` (10,000 records) and
`data/prism_community_stats.csv`. All results are reproducible with `numpy` seed 42.

**Runtime:** approximately 3–5 minutes on a standard laptop.

### Step 2: Run all experiments

```bash
python src/run_experiments.py            # full run, seed 42
python src/run_experiments.py --quick    # faster check
```

On first run this adds community `group_id`s and writes
`data/prism_dataset_with_groups.csv`. It then runs the baseline comparison
(GroupKFold by Louvain community, seed 42), temporal holdout (train months 1-4, test month 6),
ablation, robustness sweep (5-30% fraud), per-vector, economic, learned-weight
and sensitivity analyses, and writes figures to `figures/` and LaTeX tables to
`docs/`. The console log of the reported run is `docs/run_experiments_v3_output.txt`.

---

## Dataset Description

The synthetic dataset (`prism_synthetic_dataset.csv`) contains 10,000 referral
events simulated over a 180-day window (1 January – 30 June 2024).

| Column | Description |
|--------|-------------|
| `event_id` | Unique event identifier |
| `referrer_id` | Referrer account ID |
| `referee_id` | Referee account ID |
| `referrer_phone_hash` | SHA-256 hash of referrer phone number |
| `referee_phone_hash` | SHA-256 hash of referee phone number |
| `same_device` | Boolean: referrer and referee share device fingerprint |
| `same_ip_subnet` | Boolean: referrer and referee share /16 IP subnet |
| `referee_account_age` | Days since referee account registration |
| `order_value_ghs` | Order value in Ghana Cedis |
| `meets_threshold` | Boolean: order ≥ GHS 100 |
| `returned` | Boolean: order was returned |
| `returned_in_window` | Boolean: returned within 14-day window |
| `both_discounts` | Boolean: referral + promotional discount co-applied |
| `monthly_referral_count` | Referrer's qualifying referrals this calendar month |
| `in_ring_cluster` | Boolean: accounts flagged by graph ring detector |
| `credit_requested_ghs` | Referral credit amount requested |
| `public_code_exposure` | Boolean: velocity spike on referral code |
| `betweenness_centrality` | Graph betweenness centrality of referrer node |
| `pagerank` | Graph PageRank of referrer node |
| `clustering_coeff` | Graph clustering coefficient of referrer node |
| `fraud_label` | Ground truth: 0 = legitimate, 1 = fraud |
| `fraud_type` | Fraud vector: E1–E8 or 'legitimate' |
| `prism_score` | PRISM composite fraud score (≤ 0 = hold) |
| `prism_decision` | 'hold_review' or 'auto_release' |

**No real personal data is included.** All phone numbers are synthetically
generated Ghana-format numbers and stored as SHA-256 hashes. All account IDs
are synthetic.

---

## Fraud Vector Summary

| Vector | Description | Count | % of Fraud |
|--------|-------------|-------|------------|
| E1 | Self-referral | 400 | 20.0% |
| E2 | Fake account farming | 300 | 15.0% |
| E3 | Micro-order farming | 350 | 17.5% |
| E4 | Threshold + return | 250 | 12.5% |
| E5 | Credit stacking | 200 | 10.0% |
| E6 | Discount compounding | 150 | 7.5% |
| E7 | Referral ring | 180 | 9.0% |
| E8 | Code exposure | 170 | 8.5% |
| — | Legitimate | 8,000 | — |

---

## Key Results

GroupKFold cross-validation by community (no fraud ring spans train and
validation folds), N = 10,000, fraud rate = 20%. All data is synthetic.

| Method | Precision | Recall | F1 | AUC |
|--------|-----------|--------|----|-----|
| PRISM (Rule-based) | 0.705 | 0.817 | 0.757 | 0.846 |
| PRISM-LR (Learned) | 0.746 | 0.845 | 0.792 | 0.955 |
| Random Forest | 0.902 | 0.905 | 0.903 | 0.990 |
| XGBoost | 0.871 | 0.918 | 0.894 | 0.990 |
| Isolation Forest | 0.686 | 0.687 | 0.686 | 0.907 |
| One-Class SVM | 0.784 | 0.807 | 0.795 | 0.922 |

Temporal holdout (train days 1-120, test after day 150): PRISM-LR F1 0.758
(recall 0.739, down from 0.845 in cross-validation), PRISM rule-based 0.623,
XGBoost 0.861. PRISM-LR loses recall under this shift, which is a limitation.

Ring detection: Louvain on the reciprocal referral subgraph recovers all
injected E7 ring events (recall 1.000) and flags 34 legitimate events. The
synthetic rings are idealised reciprocal cliques, so real rings are likely
harder to detect.

PRISM is not intended to outperform supervised ensembles on F1; it targets
cold-start, low-label, policy-constrained deployment.

---

## Citation

If you use this dataset, code, or framework, please cite:

```bibtex
@article{Mahmud2026PRISM,
  author  = {Mahmud, Tonon Abdul-Shafiu and Abdul-Salaam, Gaddafi},
  title   = {{PRISM}: A Fraud-Resistant Design Framework for
             Mobile-First Referral Programmes in Developing Markets},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

---

## Licence

The dataset and code are released under a **Creative Commons Attribution 4.0
International (CC BY 4.0)** licence. You are free to share and adapt the
material for any purpose, provided appropriate credit is given.

---

## Contact

For questions about the dataset, code, or framework, contact:
**tasmahmud@knust.edu.gh**
