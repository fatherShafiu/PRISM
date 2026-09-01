"""
PRISM Scorer
=============
Production-ready implementation of the PRISM fraud scoring framework.

This module can be used standalone (rule-based) or with learned weights
(PRISM-LR) by passing a trained sklearn LogisticRegression model.

Usage — rule-based
------------------
    from prism_scorer import PRISMScorer

    scorer = PRISMScorer()
    event = {
        "phone_verified":         True,
        "meets_threshold":        True,
        "returned_in_window":     False,
        "referee_account_age":    3,        # days
        "same_device":            True,
        "same_ip_subnet":         True,
        "both_discounts":         False,
        "monthly_referral_count": 8,
        "in_ring_cluster":        False,
        "credit_requested_ghs":   20.0,
        "public_code_exposure":   False,
    }
    result = scorer.evaluate(event)
    print(result)
    # {'score': -65, 'decision': 'hold_review', 'triggered_signals': [...]}

Usage — PRISM-LR (learned weights)
-----------------------------------
    from sklearn.linear_model import LogisticRegression
    import joblib

    model = joblib.load("prism_lr_model.pkl")
    scorer = PRISMScorer(lr_model=model)
    result = scorer.evaluate(event)

Operator Calibration Pathway (Section 8.2 of the paper)
---------------------------------------------------------
Stage 1 (weeks 1–4):  Deploy rule-based PRISM. Log all HOLD decisions
                       without voiding credits. Accumulate labels.
Stage 2 (weeks 5–8):  Fit PRISM-LR on accumulated labels.
Stage 3 (ongoing):    Monitor false-positive rate monthly. Retrain quarterly
                       or when fraud rate shifts by >5 percentage points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Scoring constants ──────────────────────────────────────────────────────────

MAX_CREDIT_PER_ORDER = 50    # GHS
MAX_REFERRALS_MONTH  = 15    # qualifying referrals per calendar month

DEFAULT_WEIGHTS = {
    "account_age":    -30,   # referee_account_age < 7 days
    "same_device":    -40,   # shared device fingerprint
    "both_discounts": -20,   # referral + promotional discount co-applied
    "monthly_cap":    -80,   # monthly referral count > 15
    "ring_cluster":   -60,   # flagged by graph ring detector
    "credit_cap":     -45,   # credit requested > MAX_CREDIT_PER_ORDER
    "code_exposure":  -35,   # public code velocity spike detected
    "compound":       -25,   # compound penalty: same device AND same IP
}


@dataclass
class PRISMResult:
    """Result of a PRISM credit-release evaluation."""
    score:             int
    decision:          str              # "auto_release" | "hold_review" | "block"
    triggered_signals: list[str] = field(default_factory=list)
    hard_gate:         str | None = None  # which hard gate fired, if any
    lr_probability:    float | None = None  # PRISM-LR fraud probability

    @property
    def is_fraud_flagged(self) -> bool:
        return self.decision in ("hold_review", "block")

    def __str__(self) -> str:
        lines = [
            f"Decision:  {self.decision.upper()}",
            f"Score:     {self.score}",
        ]
        if self.hard_gate:
            lines.append(f"Hard gate: {self.hard_gate}")
        if self.triggered_signals:
            lines.append(f"Signals:   {', '.join(self.triggered_signals)}")
        if self.lr_probability is not None:
            lines.append(f"LR prob:   {self.lr_probability:.3f}")
        return "\n".join(lines)


class PRISMScorer:
    """
    PRISM credit-release decision engine.

    Parameters
    ----------
    weights : dict, optional
        Custom scoring weights. Defaults to the rule-based weights from
        Table 4 of the paper. Keys:
            account_age, same_device, both_discounts, monthly_cap,
            ring_cluster, credit_cap, code_exposure, compound
    lr_model : sklearn estimator, optional
        Fitted LogisticRegression (or compatible) model for PRISM-LR.
        When provided, the LR probability is computed alongside the
        rule-based score. The decision uses the LR probability if
        lr_threshold is set; otherwise the rule-based score governs.
    lr_threshold : float, optional
        Fraud probability threshold for PRISM-LR decisions (default: 0.5).
    max_credit_per_order : float
        Hard cap on credit redeemable per order (default: GHS 50).
    max_referrals_month : int
        Hard cap on qualifying referrals per month (default: 15).
    """

    def __init__(
        self,
        weights: dict[str, int] | None = None,
        lr_model=None,
        lr_threshold: float = 0.50,
        max_credit_per_order: float = MAX_CREDIT_PER_ORDER,
        max_referrals_month:  int   = MAX_REFERRALS_MONTH,
    ):
        self.weights              = weights or DEFAULT_WEIGHTS.copy()
        self.lr_model             = lr_model
        self.lr_threshold         = lr_threshold
        self.max_credit_per_order = max_credit_per_order
        self.max_referrals_month  = max_referrals_month

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, event: dict[str, Any]) -> PRISMResult:
        """
        Evaluate a single referral event and return a credit-release decision.

        Parameters
        ----------
        event : dict
            Referral event features. Required keys:
              phone_verified, meets_threshold, returned_in_window,
              referee_account_age, same_device, same_ip_subnet,
              both_discounts, monthly_referral_count, in_ring_cluster,
              credit_requested_ghs, public_code_exposure

        Returns
        -------
        PRISMResult
        """
        # Phase 1: Hard gates
        gate = self._check_hard_gates(event)
        if gate:
            return PRISMResult(
                score=-100, decision="block",
                hard_gate=gate,
            )

        # Phase 2: Soft-signal scoring
        score, signals = self._compute_score(event)

        # Phase 3: Decision
        if self.lr_model is not None:
            lr_prob = self._lr_probability(event)
            decision = "hold_review" if lr_prob >= self.lr_threshold else "auto_release"
        else:
            lr_prob = None
            decision = "hold_review" if score < 0 else "auto_release"

        return PRISMResult(
            score=score,
            decision=decision,
            triggered_signals=signals,
            lr_probability=lr_prob,
        )

    def evaluate_batch(self, events: list[dict[str, Any]]) -> list[PRISMResult]:
        """Evaluate a list of referral events."""
        return [self.evaluate(e) for e in events]

    # ── Private methods ────────────────────────────────────────────────────────

    def _check_hard_gates(self, event: dict) -> str | None:
        """Return the name of the triggered hard gate, or None."""
        if not event.get("phone_verified", True):
            return "phone_unverified"
        if not event.get("meets_threshold", True):
            return "below_threshold"
        if event.get("returned_in_window", False):
            return "returned_in_window"
        return None

    def _compute_score(self, event: dict) -> tuple[int, list[str]]:
        """Compute weighted soft-signal score and list of triggered signals."""
        w = self.weights
        score   = 0
        signals = []

        if event.get("referee_account_age", 999) < 7:
            score += w["account_age"]
            signals.append(f"account_age<7d ({w['account_age']})")

        if event.get("same_device", False):
            score += w["same_device"]
            signals.append(f"same_device ({w['same_device']})")

        if event.get("both_discounts", False):
            score += w["both_discounts"]
            signals.append(f"both_discounts ({w['both_discounts']})")

        if event.get("monthly_referral_count", 0) > self.max_referrals_month:
            score += w["monthly_cap"]
            signals.append(f"monthly_cap_exceeded ({w['monthly_cap']})")

        if event.get("in_ring_cluster", False):
            score += w["ring_cluster"]
            signals.append(f"ring_cluster ({w['ring_cluster']})")

        if event.get("credit_requested_ghs", 0) > self.max_credit_per_order:
            score += w["credit_cap"]
            signals.append(f"credit_cap_exceeded ({w['credit_cap']})")

        if event.get("public_code_exposure", False):
            score += w["code_exposure"]
            signals.append(f"code_exposure ({w['code_exposure']})")

        if event.get("same_ip_subnet", False) and event.get("same_device", False):
            score += w["compound"]
            signals.append(f"compound_co-location ({w['compound']})")

        return score, signals

    def _lr_probability(self, event: dict) -> float:
        """Compute PRISM-LR fraud probability from the fitted model."""
        FEATURE_ORDER = [
            "phone_verified", "meets_threshold", "returned_in_window",
            "referee_account_age", "same_device", "same_ip_subnet",
            "both_discounts", "monthly_referral_count", "in_ring_cluster",
            "credit_requested_ghs", "public_code_exposure",
            "stacked_credits",
            "betweenness_centrality", "pagerank", "clustering_coeff",
        ]
        import numpy as np
        X = np.array([[float(event.get(f, 0)) for f in FEATURE_ORDER]])
        return float(self.lr_model.predict_proba(X)[0, 1])

    # ── Calibration helpers ────────────────────────────────────────────────────

    @classmethod
    def from_dataset(cls, df, **kwargs) -> "PRISMScorer":
        """
        Fit a PRISM-LR model on a labelled dataset and return a PRISMScorer.

        Parameters
        ----------
        df : pd.DataFrame
            Labelled referral event dataset with column 'fraud_label'.
        **kwargs
            Passed to PRISMScorer constructor.

        Returns
        -------
        PRISMScorer with lr_model fitted.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        FEATURE_COLS = [
            "phone_verified", "meets_threshold", "returned_in_window",
            "referee_account_age", "same_device", "same_ip_subnet",
            "both_discounts", "monthly_referral_count", "in_ring_cluster",
            "credit_requested_ghs", "public_code_exposure",
            "stacked_credits",
            "betweenness_centrality", "pagerank", "clustering_coeff",
        ]
        X = df[FEATURE_COLS].fillna(0).astype(float)
        y = df["fraud_label"].values

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=1000, solver="lbfgs", random_state=42,
            )),
        ])
        pipeline.fit(X, y)
        return cls(lr_model=pipeline, **kwargs)

    def save_model(self, path: str) -> None:
        """Save the fitted PRISM-LR model to disk (requires joblib)."""
        import joblib
        if self.lr_model is None:
            raise ValueError("No LR model to save. Call from_dataset() first.")
        joblib.dump(self.lr_model, path)
        print(f"✓ Model saved → {path}")

    @classmethod
    def load_model(cls, path: str, **kwargs) -> "PRISMScorer":
        """Load a previously saved PRISM-LR model."""
        import joblib
        model = joblib.load(path)
        return cls(lr_model=model, **kwargs)
