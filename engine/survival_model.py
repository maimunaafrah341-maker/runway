"""
Invoice payment-timing model for Runway (survival analysis).

Each invoice is a time-to-event observation: the event is "payment received".
Unpaid invoices are right-censored at the evaluation date.

Time axis
---------
Time is measured in days relative to the invoice due date, shifted so that every
duration is strictly positive (lifelines' parametric models require t > 0):

    t = delay_days + offset_,   offset_ = 1 - (earliest observed payment, in days past due)

so t = 1 is the earliest any invoice has ever been paid relative to its due date.
Measuring from the due date (rather than the issue date) keeps the payment spike
around the due date aligned across 30/45/60-day terms, which is much closer to
proportional hazards than a time axis that starts at issue.

Censored invoices that are censored before t = 1 (issued recently, nowhere near
due) carry no information for the Cox partial likelihood and are dropped from the fit.

Models
------
Primary: lifelines.CoxPHFitter. If it fails to converge (ConvergenceError,
ConvergenceWarning, or a singular Hessian) the model falls back to
lifelines.WeibullAFTFitter. `model_type_` and `fallback_reason_` record what happened.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

try:
    from data.schemas import CUSTOMER_SEGMENTS
except ImportError:  # executed as `python engine/survival_model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.schemas import CUSTOMER_SEGMENTS

DURATION_COL = "duration"
EVENT_COL = "event"
REFERENCE_SEGMENT = CUSTOMER_SEGMENTS[0]  # dropped from the one-hot to avoid collinearity
SEGMENT_DUMMIES = [f"seg_{s.lower().replace('-', '_').replace(' ', '_')}" for s in CUSTOMER_SEGMENTS[1:]]
FEATURE_COLUMNS: List[str] = ["log_amount", *SEGMENT_DUMMIES, "payment_terms_days", "past_avg_delay", "is_disputed"]
_S_EVAL_EPS = 1e-9


class InvoiceSurvivalModel:
    """Predicts when open invoices will be paid.

    Args:
        prefer: "cox" (default; falls back to Weibull AFT on failure) or "weibull"
            (skip Cox entirely).
        penalizer: L2 penalty for both fitters; a small value keeps Cox stable when
            a segment has few invoices.
    """

    def __init__(self, prefer: str = "cox", penalizer: float = 0.01) -> None:
        if prefer not in ("cox", "weibull"):
            raise ValueError("prefer must be 'cox' or 'weibull'")
        self.prefer = prefer
        self.penalizer = penalizer
        self.model_ = None
        self.model_type_: Optional[str] = None
        self.fallback_reason_: Optional[str] = None
        self.offset_: Optional[int] = None
        self.segment_prior_delay_: dict = {}
        self.global_prior_delay_: float = 0.0

    # ------------------------------------------------------------------ #
    # Feature pipeline
    # ------------------------------------------------------------------ #
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        unknown = set(df["customer_segment"]) - set(CUSTOMER_SEGMENTS)
        if unknown:
            raise ValueError(f"Unknown customer_segment values: {sorted(unknown)}")
        X = pd.DataFrame(index=df.index)
        X["log_amount"] = np.log1p(df["amount"].astype(float))
        for seg, col in zip(CUSTOMER_SEGMENTS[1:], SEGMENT_DUMMIES):
            X[col] = (df["customer_segment"] == seg).astype(float)
        X["payment_terms_days"] = df["payment_terms_days"].astype(float)
        # Cold start: no payment history yet -> segment-level empirical prior.
        prior = df["customer_segment"].map(self.segment_prior_delay_).fillna(self.global_prior_delay_)
        X["past_avg_delay"] = df["customer_past_avg_delay"].astype(float).fillna(prior)
        X["is_disputed"] = df["is_disputed"].astype(float)
        return X[FEATURE_COLUMNS]

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, historical_df: pd.DataFrame) -> "InvoiceSurvivalModel":
        """Fit on paid invoices (events) plus invoices still unpaid (right-censored).

        Requires columns: delay_days, event_observed, customer_segment, amount,
        payment_terms_days, customer_past_avg_delay, is_disputed.
        """
        df = historical_df.copy()
        event = df["event_observed"].astype(bool)
        delay = df["delay_days"].astype(float)
        if not event.any():
            raise ValueError("historical_df has no paid invoices to learn from")

        # Empirical priors for cold-start customers, learned from realised delays.
        self.segment_prior_delay_ = delay[event].groupby(df.loc[event, "customer_segment"]).mean().to_dict()
        self.global_prior_delay_ = float(delay[event].mean())

        self.offset_ = int(1 - delay[event].min())
        train = self._features(df)
        train[DURATION_COL] = delay + self.offset_
        train[EVENT_COL] = event.astype(int)
        train = train[train[DURATION_COL] >= 1]  # censored before any payment can occur: no information

        self.model_, self.model_type_, self.fallback_reason_ = None, None, None
        if self.prefer == "cox":
            try:
                self.model_ = self._fit_cox(train)
                self.model_type_ = "cox"
            except (ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError) as exc:
                self.fallback_reason_ = f"{type(exc).__name__}: {exc}".splitlines()[0]
        if self.model_ is None:
            self.model_ = WeibullAFTFitter(penalizer=self.penalizer).fit(
                train, duration_col=DURATION_COL, event_col=EVENT_COL
            )
            self.model_type_ = "weibull_aft"
        return self

    def _fit_cox(self, train: pd.DataFrame) -> CoxPHFitter:
        with warnings.catch_warnings():
            # lifelines reports non-convergence as a warning; promote it so we can fall back.
            warnings.simplefilter("error", ConvergenceWarning)
            return CoxPHFitter(penalizer=self.penalizer).fit(
                train, duration_col=DURATION_COL, event_col=EVENT_COL
            )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predicting")

    def _days_past_due(self, df: pd.DataFrame, as_of_date: Optional[str]) -> np.ndarray:
        if as_of_date is not None:
            due = pd.to_datetime(df["due_date"])
            return (pd.Timestamp(as_of_date) - due).dt.days.to_numpy(dtype=float)
        if "delay_days" not in df:
            raise ValueError("Pass as_of_date or provide delay_days (days past due at evaluation)")
        return df["delay_days"].to_numpy(dtype=float)

    def survival_on_grid(self, df: pd.DataFrame, max_t: int) -> np.ndarray:
        """S(t) for t = 0..max_t on the model's time axis; shape (n, max_t + 1)."""
        self._check_fitted()
        times = np.arange(0, max_t + 1, dtype=float)
        X = self._features(df.reset_index(drop=True))
        sf = self.model_.predict_survival_function(X, times=times)
        S = sf.to_numpy().T  # lifelines returns (times, n)
        S[:, 0] = 1.0        # nothing can be paid before t = 1
        return np.clip(np.minimum.accumulate(S, axis=1), 0.0, 1.0)

    def predict_cumulative_payment_prob(
        self,
        open_invoices_df: pd.DataFrame,
        max_horizon_days: int = 60,
        as_of_date: Optional[str] = None,
    ) -> np.ndarray:
        """Conditional payment CDF for open invoices.

        Returns an array of shape (n_invoices, max_horizon_days) where [i, d] is
        P(invoice i is paid within d + 1 days after the evaluation date | unpaid at
        the evaluation date), i.e. column 0 is "by tomorrow" and the last column is
        "by day max_horizon_days".

            P(T <= t_eval + k | T > t_eval) = [S(t_eval) - S(t_eval + k)] / S(t_eval)

        The evaluation point t_eval comes from `delay_days` (days past due at the
        evaluation date, as written by data/generator.py for censored invoices) or,
        if `as_of_date` is given, from `as_of_date - due_date`.

        Invoices the model thinks should almost surely have been paid already
        (S(t_eval) ~ 0, i.e. beyond the observed delay range) get a zero row: the
        model has no evidence they will be paid within the horizon.
        """
        self._check_fitted()
        if len(open_invoices_df) == 0:
            return np.zeros((0, max_horizon_days))

        # Model-time of the evaluation date; negative for invoices not yet near their due
        # date. Lookups are clipped at t = 0 where S = 1 (no payment possible yet), but the
        # horizon is NOT shifted: day k after evaluation is model-time t_eval + k.
        t_eval = (self._days_past_due(open_invoices_df, as_of_date) + self.offset_).astype(int)
        max_t = max(int(t_eval.max()) + max_horizon_days, 1)
        S = self.survival_on_grid(open_invoices_df, max_t)

        rows = np.arange(len(open_invoices_df))[:, None]
        cols = np.clip(t_eval[:, None] + np.arange(1, max_horizon_days + 1)[None, :], 0, max_t)
        s_eval = S[np.arange(len(t_eval)), np.clip(t_eval, 0, max_t)][:, None]
        s_future = S[rows, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.where(s_eval > _S_EVAL_EPS, (s_eval - s_future) / s_eval, 0.0)
        return np.clip(cdf, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    @property
    def concordance_index_(self) -> float:
        self._check_fitted()
        return float(self.model_.concordance_index_)

    def print_summary(self) -> None:
        self._check_fitted()
        self.model_.print_summary(decimals=3)


if __name__ == "__main__":
    ledger_path = Path(__file__).resolve().parents[1] / "data" / "sample_ledger.csv"
    if not ledger_path.exists():
        raise SystemExit(f"{ledger_path} not found - run `python data/generator.py` first")
    ledger = pd.read_csv(ledger_path, parse_dates=["issue_date", "due_date", "actual_payment_date"])

    model = InvoiceSurvivalModel().fit(ledger)
    print(f"Model: {model.model_type_}" + (f" (fallback: {model.fallback_reason_})" if model.fallback_reason_ else ""))
    print(f"Time offset: t = days past due + {model.offset_}")
    print("Cold-start priors (mean days past due):",
          {k: round(v, 1) for k, v in model.segment_prior_delay_.items()})
    model.print_summary()
    print(f"\nConcordance index: {model.concordance_index_:.3f}")

    open_inv = ledger[~ledger["event_observed"].astype(bool)].reset_index(drop=True)
    probs = model.predict_cumulative_payment_prob(open_inv, max_horizon_days=60)

    # --- self-test assertions ---
    assert probs.shape == (len(open_inv), 60)
    assert np.all((probs >= 0) & (probs <= 1)), "probabilities out of [0, 1]"
    assert np.all(np.diff(probs, axis=1) >= -1e-12), "CDF must be non-decreasing"
    assert model.concordance_index_ > 0.5, "model is no better than random ordering"

    cold = ledger.copy()
    cold["customer_past_avg_delay"] = np.nan
    assert not model._features(cold).isna().any().any(), "cold-start imputation left NaNs"

    weibull = InvoiceSurvivalModel(prefer="weibull").fit(ledger)
    wprobs = weibull.predict_cumulative_payment_prob(open_inv)
    assert weibull.model_type_ == "weibull_aft" and wprobs.shape == probs.shape
    print(f"Weibull AFT fallback path OK (concordance {weibull.concordance_index_:.3f})")

    # Conditioning check: an invoice already 30 days overdue vs. one not yet due.
    probe = open_inv.iloc[[0, 0]].copy()
    probe["delay_days"] = [-10, 30]
    p_probe = model.predict_cumulative_payment_prob(probe, 60)
    print(f"Same invoice, P(paid within 30d): due in 10 days -> {p_probe[0, 29]:.2f}, "
          f"30 days overdue -> {p_probe[1, 29]:.2f}")
    # Regression: an invoice due beyond (horizon + earliest-ever early payment) cannot be paid
    # within the horizon; its curve must not be pulled forward.
    far = open_inv.iloc[[0]].copy()
    far["delay_days"] = -(model.offset_ + 30)
    assert model.predict_cumulative_payment_prob(far, 30)[0, -1] == 0.0, "not-yet-due curve shifted forward"
    print("All self-test assertions passed.")

    # --- sample curves ---
    print("\nSample conditional payment curves (open invoices):")
    sample_idx = (
        open_inv.assign(i=range(len(open_inv)))
        .sort_values("amount", ascending=False)
        .groupby("customer_segment").head(1)["i"].tolist()
    )
    checkpoints = [7, 14, 30, 45, 60]
    header = f"{'invoice':<15}{'segment':<20}{'amount':>11}{'days past due':>15}" + "".join(f"{'P@' + str(d) + 'd':>8}" for d in checkpoints)
    print(header)
    for i in sample_idx:
        r = open_inv.iloc[i]
        cells = "".join(f"{probs[i, d - 1]:>8.2f}" for d in checkpoints)
        print(f"{r.invoice_id:<15}{r.customer_segment:<20}{r.amount:>11,.0f}{int(r.delay_days):>15}{cells}")
