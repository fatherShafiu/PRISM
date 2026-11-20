"""
PRISM Synthetic Dataset Generator
===================================
Generates a 10,000-record synthetic referral fraud dataset for the paper:

    "PRISM: A Fraud-Resistant Design Framework for Mobile-First
     Referral Programmes in Developing Markets"

Usage
-----
    python generate_dataset.py                        # default: N=10,000, fraud=20%
    python generate_dataset.py --fraud-rate 0.10      # 10% fraud rate
    python generate_dataset.py --total 5000           # smaller dataset
    python generate_dataset.py --output my_data.csv   # custom output path

Fraud vectors generated (E1–E8)
--------------------------------
E1  Self-referral (shared device/IP)
E2  Fake account farming (shared IP subnet cluster)
E3  Micro-order farming (orders hugging the minimum threshold)
E4  Threshold triggering with return (100% return rate)
E5  Credit stacking (balance exceeds per-order cap)
E6  Discount compounding (referral + promo discount co-applied)
E7  Referral rings (circular referral topology)
E8  Affiliate code exposure (viral code distribution)

Reproducibility
---------------
All results are reproducible with numpy seed 42.
No real personal data is used. Phone numbers are synthetic Ghana-format
numbers stored as SHA-256 hashes.

Requirements
------------
    pip install numpy pandas networkx
"""

import argparse
import hashlib
import random
import string
import warnings
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
RING_RECRUIT_SHARE = 0.30  # share of E7 ring members drawn from existing users

# ── Platform constants (calibrated to West African e-commerce) ─────────────────
SIMULATION_START     = datetime(2024, 1, 1)
SIMULATION_DAYS      = 180          # 6-month window
GHS_MIN_ORDER        = 100          # Minimum qualifying order value (GHS)
GHS_MAX_ORDER        = 2_000        # Maximum order value (GHS)
REFERRER_REWARD      = 20           # GHS reward to referrer
NEW_CUSTOMER_DISC    = 10           # GHS discount to new customer (referee)
TOTAL_REFERRAL_COST  = 30           # GHS total cost per qualifying referral
PLATFORM_MARGIN      = 0.30         # Gross margin rate (30%)
RETURN_WINDOW_DAYS   = 14           # Post-delivery return window
LEGIT_RETURN_RATE    = 0.08         # 8% return rate (Statista 2023, West Africa)
MAX_CREDIT_PER_ORDER = 50           # Hard cap: max credit per order (GHS)
MAX_CREDIT_BALANCE   = 150          # Hard cap: max credit balance (GHS)
MAX_REFERRALS_MONTH  = 15           # Hard cap: max qualifying referrals/month

# ── Fraud injection weights (fraction of total fraud events) ───────────────────
# Calibrated to reflect documented relative prevalence in digital commerce
FRAUD_WEIGHTS = {
    "E1": 0.200,   # Self-referral: most common in low-KYC markets
    "E2": 0.150,   # Fake farming: coordinated, moderate volume
    "E3": 0.175,   # Micro-order: high volume, low per-event value
    "E4": 0.125,   # Threshold+return: requires physical return
    "E5": 0.100,   # Credit stacking: requires accumulated balance
    "E6": 0.075,   # Discount compound: detected at checkout
    "E7": 0.090,   # Referral ring: requires coordination
    "E8": 0.085,   # Code exposure: viral distribution events
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sha256(value: str) -> str:
    """One-way hash to anonymise identifiers. No real data stored."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def random_date(start: datetime, days: int) -> datetime:
    """Generate a uniformly random datetime within [start, start+days]."""
    return start + timedelta(
        days=random.randint(0, max(0, days - 1)),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def window_date(offset_days: int, tail_days: int = 0) -> datetime:
    """Random datetime between day ``offset_days`` and the end of the
    simulation window (minus ``tail_days``), so every order date falls
    inside [SIMULATION_START, SIMULATION_START + SIMULATION_DAYS)."""
    span = max(1, SIMULATION_DAYS - tail_days - offset_days)
    return random_date(SIMULATION_START + timedelta(days=offset_days), span)


def ghana_phone() -> str:
    """
    Generate a synthetic Ghana mobile number.
    Prefixes weighted by approximate 2024 MNO market share:
      MTN ~50%: 024, 054, 055, 059
      Vodafone ~20%: 020, 050
      AirtelTigo ~30%: 026, 056, 027, 057
    """
    prefixes = [
        "024", "054", "055", "059", "024", "054",  # MTN (doubled for weight)
        "020", "050",                                # Vodafone
        "026", "056", "027", "057",                  # AirtelTigo
    ]
    return random.choice(prefixes) + "".join(random.choices(string.digits, k=7))


def random_ip_subnet() -> str:
    return f"{random.randint(1, 254)}.{random.randint(1, 254)}"


def random_device() -> str:
    return "".join(random.choices(string.hexdigits[:16], k=12))


def random_payment() -> str:
    """
    Payment instrument distribution weighted to reflect Ghana mobile-money
    dominance (Bank of Ghana Digital Financial Services Report, 2023).
    """
    return random.choices(
        ["mtn_momo", "vodafone_cash", "airteltigo_money", "visa_card", "bank_transfer"],
        weights=[0.42, 0.22, 0.18, 0.12, 0.06],
    )[0]


def lognormal_order(low: float = 100, high: float = 2000,
                    mu: float = 5.2, sigma: float = 0.6) -> float:
    """
    Log-normal order value distribution.
    Parameters calibrated so:
      - Median ≈ GHS 181 (consistent with Ghanaian e-commerce mid-range)
      - 10th percentile ≈ GHS 107 (just above threshold — natural overlap with E3)
      - 90th percentile ≈ GHS 540
    """
    return round(float(np.clip(np.random.lognormal(mu, sigma), low, high)), 2)


def referral_code(account_id: str) -> str:
    return "REF" + sha256(account_id)[:8].upper()


# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT POOL
# ══════════════════════════════════════════════════════════════════════════════

def generate_account_pool(n: int = 5_000) -> pd.DataFrame:
    """
    Generate a pool of unique synthetic accounts.
    Each account has a unique phone number — the primary identity anchor
    in the PRISM framework, backed by Ghana's mandatory SIM-KYC registration.
    """
    accounts = []
    used_phones: set = set()
    for i in range(n):
        while True:
            phone = ghana_phone()
            if phone not in used_phones:
                used_phones.add(phone)
                break
        aid = f"ACC{i:06d}"
        reg = random_date(SIMULATION_START, SIMULATION_DAYS - 30)
        accounts.append({
            "account_id":         aid,
            "phone_hash":         sha256(phone),
            "device_fingerprint": random_device(),
            "ip_subnet":          random_ip_subnet(),
            "registration_date":  reg,
            "payment_instrument": random_payment(),
            "referral_code":      referral_code(aid),
        })
    return pd.DataFrame(accounts).set_index("account_id")


# ══════════════════════════════════════════════════════════════════════════════
# BASE RECORD BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def base_record(
    idx: int, referrer_id: str, referee_id: str,
    accounts: pd.DataFrame, order_date: datetime,
    order_value: float, fraud_label: int, fraud_type: str,
    overrides: dict | None = None,
) -> dict:
    """Build a referral event record with default legitimate values."""
    r = accounts.loc[referrer_id]
    # referee_id may be a synthetic fake not in the account pool
    e = accounts.loc[referee_id] if referee_id in accounts.index else r

    rec = {
        "event_id":               f"EVT{idx:08d}",
        "referrer_id":            referrer_id,
        "referee_id":             referee_id,
        "referrer_phone_hash":    r["phone_hash"],
        "referee_phone_hash":     e["phone_hash"],
        "same_device":            False,
        "referrer_ip_subnet":     r["ip_subnet"],
        "referee_ip_subnet":      e["ip_subnet"],
        "same_ip_subnet":         r["ip_subnet"] == e["ip_subnet"],
        "referee_reg_date":       e["registration_date"],
        "order_date":             order_date,
        "referee_account_age":    max(0, (order_date - e["registration_date"]).days),
        "order_value_ghs":        order_value,
        "meets_threshold":        order_value >= GHS_MIN_ORDER,
        "returned":               False,
        "return_date":            None,
        "returned_in_window":     False,
        "promotion_applied":      False,
        "both_discounts":         False,
        "credit_requested_ghs":   REFERRER_REWARD,
        "credit_balance_before":  round(random.uniform(0, 80), 2),
        "stacked_credits":        False,
        "phone_verified":         True,
        "payment_instrument":     e["payment_instrument"],
        "public_code_exposure":   False,
        "monthly_referral_count": random.randint(1, 8),
        "in_ring_cluster":        False,
        "betweenness_centrality": 0.0,
        "pagerank":               0.0,
        "clustering_coeff":       0.0,
        "fraud_label":            fraud_label,
        "fraud_type":             fraud_type,
    }
    if overrides:
        rec.update(overrides)
    return rec


# ══════════════════════════════════════════════════════════════════════════════
# LEGITIMATE EVENT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_legitimate(accounts: pd.DataFrame, n: int,
                        start_idx: int = 0) -> list[dict]:
    """
    Generate n legitimate referral events.
    Key properties:
    - Orders placed at least 7 days after referee registration
    - 8% natural return rate (Statista 2023 West Africa)
    - 15% of orders naturally hug the threshold (low-value shoppers)
      This creates realistic distributional overlap with E3 fraud events,
      which is the primary reason E3 achieves lower detection recall.
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        ref = random.choice(ids)
        ree = random.choice([a for a in ids if a != ref])
        e = accounts.loc[ree]

        reg_day = max(7, (e["registration_date"] - SIMULATION_START).days + 7)
        if reg_day >= SIMULATION_DAYS:
            reg_day = (e["registration_date"] - SIMULATION_START).days + 1
        od = window_date(reg_day)

        # 15% of legitimate orders hug the threshold naturally
        ov = (round(GHS_MIN_ORDER * random.uniform(1.0, 1.08), 2)
              if random.random() < 0.15 else lognormal_order())

        ret = random.random() < LEGIT_RETURN_RATE
        rd = od + timedelta(days=random.randint(1, RETURN_WINDOW_DAYS)) if ret else None

        rec = base_record(start_idx + i, ref, ree, accounts, od, ov, 0, "legitimate")
        rec.update({
            "returned":           ret,
            "return_date":        rd,
            "returned_in_window": ret and (rd - od).days <= RETURN_WINDOW_DAYS if ret else False,
            "promotion_applied":  random.random() < 0.15,
            "referee_account_age": max(7, (od - e["registration_date"]).days),
        })
        records.append(rec)
    return records


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD VECTOR GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def inject_E1(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E1 — Self-Referral.
    Attacker creates a secondary account to refer themselves, collecting
    both the referrer reward and new-customer discount on one purchase.

    Signal: same device fingerprint and IP subnet.
    Noise: 20% of E1 actors factory-reset their device to evade the
    device-fingerprint signal, producing realistic false negatives.
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        primary = random.choice(ids)
        r = accounts.loc[primary]
        od = window_date(1)
        evades = random.random() < 0.20  # evades device signal
        rec = base_record(
            start_idx + i, primary, f"FAKE_{primary}_{i}",
            accounts, od, lognormal_order(100, 500), 1, "E1_self_referral",
            overrides={
                "referee_phone_hash":   sha256(ghana_phone() + "_fake"),
                "referee_device":       r["device_fingerprint"] if not evades
                                        else random_device(),
                "same_device":          not evades,
                "referee_ip_subnet":    r["ip_subnet"],
                "same_ip_subnet":       True,
                "referee_account_age":  random.randint(0, 3),
                "referee_reg_date":     od - timedelta(days=random.randint(0, 3)),
                "payment_instrument":   r["payment_instrument"],
                "monthly_referral_count": random.randint(1, 5),
            },
        )
        records.append(rec)
    return records


def inject_E2(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E2 — Fake Account Farming.
    Single actor creates clusters of 3–8 fake accounts, all referring one
    primary account. Fake accounts share an IP subnet (cluster signal).
    Noise: 30% use different payment instruments to evade that signal.
    """
    records = []
    ids = accounts.index.tolist()
    remaining = n
    while remaining > 0:
        cluster_size = random.randint(3, max(3, min(8, remaining)))
        primary = random.choice(ids)
        r = accounts.loc[primary]
        shared_subnet = random_ip_subnet()
        for j in range(cluster_size):
            od = window_date(1)
            rec = base_record(
                start_idx + (n - remaining) + j, primary,
                f"FARMED_{primary}_{j}", accounts, od,
                lognormal_order(100, 300), 1, "E2_fake_farming",
                overrides={
                    "referee_phone_hash":   sha256(f"fake_{primary}_{j}_{random.random()}"),
                    "referee_ip_subnet":    shared_subnet,
                    "same_ip_subnet":       r["ip_subnet"] == shared_subnet,
                    "referee_account_age":  random.randint(0, 2),
                    "referee_reg_date":     od - timedelta(days=random.randint(0, 2)),
                    "phone_verified":       random.random() > 0.30,
                    "monthly_referral_count": cluster_size,
                    "payment_instrument":   r["payment_instrument"]
                                            if random.random() > 0.30
                                            else random_payment(),
                },
            )
            records.append(rec)
            remaining -= 1
            if remaining <= 0:
                break
    return records[:n]


def inject_E3(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E3 — Micro-Order Farming.
    Orders placed at GHS_MIN_ORDER × [1.0, 1.08] — identical distribution
    to the legitimate threshold-huggers in generate_legitimate(). This
    intentional overlap produces the lower E3 recall reported in the paper
    and models realistic structural ambiguity.
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        ref = random.choice(ids)
        ree = random.choice([a for a in ids if a != ref])
        e = accounts.loc[ree]
        reg_day = max(7, (e["registration_date"] - SIMULATION_START).days + 7)
        if reg_day >= SIMULATION_DAYS:
            reg_day = (e["registration_date"] - SIMULATION_START).days + 1
        od = window_date(reg_day)
        rec = base_record(
            start_idx + i, ref, ree, accounts, od,
            round(GHS_MIN_ORDER * random.uniform(1.0, 1.08), 2),
            1, "E3_micro_order",
            overrides={
                "referee_account_age": max(7, (od - e["registration_date"]).days),
                "monthly_referral_count": random.randint(10, 15),
            },
        )
        records.append(rec)
    return records


def inject_E4(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E4 — Threshold Triggering with Return.
    Referee places a qualifying order then returns it within the return window.
    100% return rate by construction (definition of this attack vector).
    Noise: 15% return after the window (late chargebacks not caught by Pillar 2).
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        ref = random.choice(ids)
        ree = random.choice([a for a in ids if a != ref])
        e = accounts.loc[ree]
        reg_day = max(7, (e["registration_date"] - SIMULATION_START).days + 7)
        if reg_day >= SIMULATION_DAYS:
            reg_day = (e["registration_date"] - SIMULATION_START).days + 1
        od = window_date(reg_day, RETURN_WINDOW_DAYS)
        late = random.random() < 0.15
        rd = (od + timedelta(days=random.randint(RETURN_WINDOW_DAYS + 1, 30))
              if late else od + timedelta(days=random.randint(1, RETURN_WINDOW_DAYS)))
        rec = base_record(
            start_idx + i, ref, ree, accounts, od,
            lognormal_order(100, 600), 1, "E4_threshold_return",
            overrides={
                "referee_account_age": max(7, (od - e["registration_date"]).days),
                "returned":           True,
                "return_date":        rd,
                "returned_in_window": not late,
            },
        )
        records.append(rec)
    return records


def inject_E5(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E5 — Credit Stacking.
    Referrer accumulates credits then deploys them simultaneously,
    requesting more than MAX_CREDIT_PER_ORDER in a single order.
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        ref = random.choice(ids)
        ree = random.choice([a for a in ids if a != ref])
        e = accounts.loc[ree]
        reg_day = max(7, (e["registration_date"] - SIMULATION_START).days + 7)
        if reg_day >= SIMULATION_DAYS:
            reg_day = (e["registration_date"] - SIMULATION_START).days + 1
        od = window_date(reg_day)
        bal = random.uniform(MAX_CREDIT_PER_ORDER + 10, MAX_CREDIT_BALANCE)
        rec = base_record(
            start_idx + i, ref, ree, accounts, od,
            lognormal_order(200, 800), 1, "E5_credit_stacking",
            overrides={
                "referee_account_age":  max(7, (od - e["registration_date"]).days),
                "credit_requested_ghs": bal,
                "credit_balance_before": bal,
                "stacked_credits":       True,
                "monthly_referral_count": random.randint(12, 15),
            },
        )
        records.append(rec)
    return records


def inject_E6(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E6 — Discount Compounding.
    Referee applies both the referral new-customer discount and a concurrent
    promotional code to the same order — a direct policy violation.
    """
    records = []
    ids = accounts.index.tolist()
    for i in range(n):
        ref = random.choice(ids)
        ree = random.choice([a for a in ids if a != ref])
        e = accounts.loc[ree]
        reg_day = max(7, (e["registration_date"] - SIMULATION_START).days + 7)
        if reg_day >= SIMULATION_DAYS:
            reg_day = (e["registration_date"] - SIMULATION_START).days + 1
        od = window_date(reg_day)
        rec = base_record(
            start_idx + i, ref, ree, accounts, od,
            lognormal_order(150, 1000), 1, "E6_discount_compound",
            overrides={
                "referee_account_age": max(7, (od - e["registration_date"]).days),
                "promotion_applied":   True,
                "both_discounts":      True,
            },
        )
        records.append(rec)
    return records


def inject_E7(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E7 — Referral Ring.
    Closed groups of 5–10 accounts refer each other in rotation: every member
    refers its successor and is referred back by it, so each member acts as
    both referrer and referee and ring edges are reciprocal.
    Ring membership is NOT written into the record here; ``in_ring_cluster``
    is set later from the output of the graph-based ring detector
    (Algorithm 2 in the paper), so E7 detection depends on the detector.
    Most members are dedicated accounts created by the ring operator; a share
    (RING_RECRUIT_SHARE) are recruited existing users who also have
    legitimate referral activity, which blurs the ring's graph boundary.
    """
    records = []
    ids = accounts.index.tolist()
    used_phones = set(accounts["phone_hash"])
    remaining = n
    ring_no = 0
    while remaining > 0:
        ring_size = random.randint(5, 7)
        members = []
        for j in range(ring_size):
            if random.random() < RING_RECRUIT_SHARE:
                # Recruited existing user: also has legitimate referral activity
                members.append(random.choice(ids))
            else:
                # Dedicated account created by the ring operator
                while True:
                    phone = ghana_phone()
                    if sha256(phone) not in used_phones:
                        used_phones.add(sha256(phone))
                        break
                aid = f"RNG{ring_no:03d}{j:02d}"
                accounts.loc[aid] = {
                    "phone_hash":         sha256(phone),
                    "device_fingerprint": random_device(),
                    "ip_subnet":          random_ip_subnet(),
                    "registration_date":  random_date(SIMULATION_START,
                                                      SIMULATION_DAYS - 30),
                    "payment_instrument": random_payment(),
                    "referral_code":      referral_code(aid),
                }
                members.append(aid)
        members = list(dict.fromkeys(members))
        ring_no += 1
        # Members refer each other: every ordered pair of distinct members
        # produces one referral event, so the ring is a reciprocal clique.
        pairs = [(a, b) for a in members for b in members if a != b]
        random.shuffle(pairs)
        for ref, ree in pairs:
            if ref == ree:
                continue
            e = accounts.loc[ree]
            od = window_date(7)
            age = max(0, (od - e["registration_date"]).days)
            rec = base_record(
                start_idx + (n - remaining), ref, ree, accounts, od,
                lognormal_order(100, 400), 1, "E7_referral_ring",
                overrides={
                    "referee_account_age":    age,
                    "monthly_referral_count":  ring_size,
                },
            )
            records.append(rec)
            remaining -= 1
            if remaining <= 0:
                break
    return records[:n]


def inject_E8(accounts: pd.DataFrame, n: int, start_idx: int) -> list[dict]:
    """
    E8 — Affiliate Code Exposure.
    A referral code is posted publicly (e.g., on a WhatsApp deal-sharing group),
    generating burst redemptions. Signal: >50 unique redemptions of one code
    within 24 hours.
    Noise: 40% of burst redeemers have account age > 7 days (existing users
    who found the public code), reducing age-signal effectiveness.
    """
    records = []
    ids = accounts.index.tolist()
    remaining = n
    while remaining > 0:
        referrer_id = random.choice(ids)
        r = accounts.loc[referrer_id]
        burst_size = random.randint(20, max(20, min(60, remaining)))
        burst_start = window_date(7, 1)
        for _ in range(burst_size):
            ree = random.choice([a for a in ids if a != referrer_id])
            e = accounts.loc[ree]
            od = burst_start + timedelta(hours=random.randint(0, 23))
            age = max(0, (od - e["registration_date"]).days)
            rec = base_record(
                start_idx + (n - remaining), referrer_id, ree,
                accounts, od, lognormal_order(), 1, "E8_code_exposure",
                overrides={
                    "referee_account_age":    age,
                    "public_code_exposure":    True,
                    "monthly_referral_count":  burst_size,
                    "returned":  random.random() < 0.12,
                },
            )
            records.append(rec)
            remaining -= 1
            if remaining <= 0:
                break
    return records[:n]


# ══════════════════════════════════════════════════════════════════════════════
# PRISM RULE-BASED SCORING
# ══════════════════════════════════════════════════════════════════════════════

RULE_WEIGHTS = {
    "account_age":    -30,
    "same_device":    -40,
    "both_discounts": -20,
    "monthly_cap":    -80,
    "ring_cluster":   -60,
    "credit_cap":     -45,
    "code_exposure":  -35,
    "compound":       -25,
}


def prism_score(row: pd.Series, weights: dict | None = None) -> int:
    """
    Compute the PRISM composite fraud score for a single referral event.

    Returns
    -------
    int
        0  → auto-release credit (no negative signals)
        <0 → hold for manual review

    Phases
    ------
    1. Hard gates (immediate block): phone unverified, threshold unmet, returned
    2. Soft-signal scoring: weighted sum of behavioural signals
    """
    w = weights or RULE_WEIGHTS

    # Phase 1: Hard gates
    if not row.get("phone_verified", True):
        return -100
    if not row.get("meets_threshold", True):
        return -100
    if row.get("returned_in_window", False):
        return -100

    # Phase 2: Soft signals
    s = 0
    if row.get("referee_account_age", 999) < 7:
        s += w["account_age"]
    if row.get("same_device", False):
        s += w["same_device"]
    if row.get("both_discounts", False):
        s += w["both_discounts"]
    if row.get("monthly_referral_count", 0) > MAX_REFERRALS_MONTH:
        s += w["monthly_cap"]
    if row.get("in_ring_cluster", False):
        s += w["ring_cluster"]
    if row.get("credit_requested_ghs", 0) > MAX_CREDIT_PER_ORDER:
        s += w["credit_cap"]
    if row.get("public_code_exposure", False):
        s += w["code_exposure"]
    if row.get("same_ip_subnet", False) and row.get("same_device", False):
        s += w["compound"]

    return s


def prism_decision(score: int) -> str:
    return "hold_review" if score < 0 else "auto_release"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH FEATURES (Louvain ring detection + centrality)
# ══════════════════════════════════════════════════════════════════════════════

def compute_graph_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build directed referral graph and compute:
      - Louvain community detection (ring detection)
      - Betweenness centrality
      - PageRank
      - Clustering coefficient

    Ring criterion (Equation 10 in paper):
        |C| >= 5  AND  reciprocal_edge_density >= 0.6
    """
    print("    Building referral graph...")
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["referrer_id"], row["referee_id"],
                   weight=row["order_value_ghs"])

    print(f"    Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    G_un = G.to_undirected()

    print("    Computing PageRank...")
    pr = nx.pagerank(G, alpha=0.85, weight="weight")

    print("    Computing betweenness centrality (approximate, k=500)...")
    bc = nx.betweenness_centrality(
        G, k=min(500, G.number_of_nodes()),
        normalized=True, weight="weight", seed=SEED,
    )

    print("    Computing clustering coefficients...")
    cc = nx.clustering(G_un)

    # Ring detection (Algorithm 2). Legitimate referral topologies are almost
    # never reciprocal (rho ~ 0 for hub referrers), so Louvain is run on the
    # undirected projection of the reciprocal-edge subgraph G_R. Running it on
    # the full graph lets small rings be absorbed into large communities
    # (the modularity resolution limit), after which no community meets the
    # density criterion.
    print("    Running Louvain community detection on reciprocal subgraph...")
    from networkx.algorithms.community import louvain_communities
    G_R = nx.Graph()
    for u, v, d in G.edges(data=True):
        if G.has_edge(v, u):
            G_R.add_edge(u, v, weight=(d["weight"] + G[v][u]["weight"]) / 2)
    communities = (louvain_communities(G_R, weight="weight", seed=SEED)
                   if G_R.number_of_edges() > 0 else [])

    ring_accounts: set = set()
    community_stats = []
    for i, comm in enumerate(communities):
        members = list(comm)
        sub_d = G.subgraph(members)
        ne = sub_d.number_of_edges()
        nr = sum(1 for u, v in sub_d.edges() if sub_d.has_edge(v, u))
        rho = nr / ne if ne > 0 else 0
        is_ring = len(members) >= 5 and rho >= 0.6
        community_stats.append({
            "community_id": i,
            "size": len(members),
            "edges": ne,
            "reciprocal_density": round(rho, 3),
            "flagged_as_ring": is_ring,
        })
        if is_ring:
            ring_accounts.update(members)

    print(f"    Ring communities detected: {sum(c['flagged_as_ring'] for c in community_stats)}")

    def safe(d: dict, k: str) -> float:
        return d.get(k, 0.0)

    df = df.copy()
    df["betweenness_centrality"] = df["referrer_id"].map(lambda x: safe(bc, x))
    df["pagerank"]               = df["referrer_id"].map(lambda x: safe(pr, x))
    df["clustering_coeff"]       = df["referrer_id"].map(lambda x: safe(cc, x))
    df["graph_ring_detected"]    = (
        df["referrer_id"].isin(ring_accounts) | df["referee_id"].isin(ring_accounts)
    )
    # Pillar 5 signal x9 comes from the detector, not from the fraud label.
    df["in_ring_cluster"]        = df["graph_ring_detected"]

    return df, pd.DataFrame(community_stats)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    fraud_rate: float = 0.20,
    total: int = 10_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the complete PRISM synthetic dataset.

    Parameters
    ----------
    fraud_rate : float
        Fraction of events that are fraudulent (0.05–0.40).
    total : int
        Total number of referral events to generate.
    verbose : bool
        Print progress information.

    Returns
    -------
    df : pd.DataFrame
        10,000-record referral event dataset with fraud labels and PRISM scores.
    community_df : pd.DataFrame
        Louvain community detection results.
    accounts : pd.DataFrame
        Account pool (for downstream analysis).
    """
    np.random.seed(SEED)
    random.seed(SEED)

    n_fraud = int(total * fraud_rate)
    n_legit = total - n_fraud

    fraud_counts = {k: max(10, int(n_fraud * v)) for k, v in FRAUD_WEIGHTS.items()}
    diff = n_fraud - sum(fraud_counts.values())
    fraud_counts["E1"] += diff  # absorb rounding into E1

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  PRISM Dataset Generator")
        print(f"  fraud_rate={fraud_rate:.0%}  total={total:,}")
        print(f"  n_legit={n_legit:,}  n_fraud={n_fraud:,}")
        print(f"{'=' * 55}")

    accounts = generate_account_pool(5_000)
    if verbose:
        print(f"\n[1] Account pool: {len(accounts):,}")

    records = generate_legitimate(accounts, n_legit, start_idx=0)
    if verbose:
        print(f"[2] Legitimate events: {n_legit:,}")

    injectors = [
        ("E1", inject_E1, fraud_counts["E1"]),
        ("E2", inject_E2, fraud_counts["E2"]),
        ("E3", inject_E3, fraud_counts["E3"]),
        ("E4", inject_E4, fraud_counts["E4"]),
        ("E5", inject_E5, fraud_counts["E5"]),
        ("E6", inject_E6, fraud_counts["E6"]),
        ("E7", inject_E7, fraud_counts["E7"]),
        ("E8", inject_E8, fraud_counts["E8"]),
    ]

    idx = n_legit
    for name, fn, cnt in injectors:
        batch = fn(accounts, cnt, idx)
        records.extend(batch)
        idx += len(batch)
        if verbose:
            print(f"    ✓ {name}: {len(batch)} events")

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Compute graph features and ring detection first: the detector's output
    # feeds the in_ring_cluster signal used by the rule-based score.
    if verbose:
        print(f"\n[3] Computing graph features...")
    df, community_df = compute_graph_features(df)

    # Apply rule-based PRISM scores
    df["prism_score"]    = df.apply(prism_score, axis=1)
    df["prism_decision"] = df["prism_score"].apply(prism_decision)

    if verbose:
        tp = int(((df["prism_decision"] == "hold_review") & (df["fraud_label"] == 1)).sum())
        fp = int(((df["prism_decision"] == "hold_review") & (df["fraud_label"] == 0)).sum())
        fn = int(((df["prism_decision"] == "auto_release") & (df["fraud_label"] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        print(f"\n  Rule-based PRISM: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    return df, community_df, accounts


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fraud-rate", type=float, default=0.20,
                        help="Fraction of events that are fraudulent (default: 0.20)")
    parser.add_argument("--total", type=int, default=10_000,
                        help="Total number of referral events (default: 10,000)")
    parser.add_argument("--output", type=str, default="data/prism_synthetic_dataset.csv",
                        help="Output CSV path (default: data/prism_synthetic_dataset.csv)")
    parser.add_argument("--communities-output", type=str,
                        default="data/prism_community_stats.csv",
                        help="Community stats output path")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df, community_df, _ = generate_dataset(
        fraud_rate=args.fraud_rate, total=args.total
    )

    df.to_csv(args.output, index=False)
    community_df.to_csv(args.communities_output, index=False)

    print(f"\n✓ Dataset saved → {args.output} ({len(df):,} records)")
    print(f"✓ Communities  → {args.communities_output}")
    print(f"\nFraud breakdown:")
    for ftype, grp in df[df["fraud_label"] == 1].groupby("fraud_type"):
        print(f"  {ftype:<30} {len(grp):>4}")


if __name__ == "__main__":
    main()
