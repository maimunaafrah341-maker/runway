"""
Monte Carlo cash-flow simulation for Runway.

Turns per-invoice payment-timing distributions (from engine/survival_model.py) into
n_simulations daily cash paths over the forecast horizon, with customer-correlated
lateness via a Gaussian copula, then attributes deficit risk to individual invoices
by counterfactual re-simulation.

Day convention
--------------
Column t (0-based) of every (paths, days) matrix is the closing balance of day t + 1
after the as-of date, matching payment_probs[:, t] = P(paid within t + 1 days).
Expenses due on or before the as-of date are treated as already paid.

Correlated lateness
-------------------
For path s, customer c and invoice i of customer c:
    Z_c ~ N(0, 1),  eps_i ~ N(0, 1)
    U_i = Phi(sqrt(rho) * Z_c + sqrt(1 - rho) * eps_i)
Invoice i arrives on the first day d with U_i <= CDF_i[d]; if U_i > CDF_i[-1] it is
not paid within the horizon. A high Z_c pushes all of customer c's invoices late
together. Each U_i is still Uniform(0, 1), so every invoice's marginal timing
distribution is exactly the survival model's.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

try:
    from data.schemas import SimulationResult
except ImportError:  # executed as `python engine/monte_carlo.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.schemas import SimulationResult

TOP_K_RISK_DRIVERS = 5
COUNTERFACTUAL_HORIZON_DAYS = 30


class CashFlowSimulator:
    """Vectorised cash-flow simulator.

    The RNG is re-seeded on every `run`, so repeated runs with modified inputs (what-if
    sliders, interventions) share the same random draws (common random numbers) and
    differences between runs reflect the change, not sampling noise.
    """

    def __init__(self, n_simulations: int = 5000, horizon_days: int = 60, rho: float = 0.5, seed: int = 42) -> None:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if horizon_days < COUNTERFACTUAL_HORIZON_DAYS:
            raise ValueError(f"horizon_days must be >= {COUNTERFACTUAL_HORIZON_DAYS}")
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.rho = rho
        self.seed = seed
        # Populated by run(), for downstream modules (optimizer, UI drill-downs).
        self.uniforms_: Optional[np.ndarray] = None       # (n_sims, n_invoices) copula draws U_i
        self.payment_probs_: Optional[np.ndarray] = None  # (n_invoices, horizon_days) CDFs used
        self.payment_days_: Optional[np.ndarray] = None   # (n_sims, n_invoices); horizon_days = not paid
        self.trajectories_: Optional[np.ndarray] = None   # (n_sims, horizon_days)
        self.outflows_: Optional[np.ndarray] = None       # (horizon_days,)
        self.open_invoice_ids_: Optional[List[str]] = None
        self.risk_table_: Optional[pd.DataFrame] = None   # counterfactual result for every open invoice

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def sample_uniforms(self, customer_ids: pd.Series) -> np.ndarray:
        """Gaussian-copula uniforms U (n_sims, n_invoices), correlated within each customer."""
        rng = np.random.default_rng(self.seed)
        cust_codes, uniques = pd.factorize(customer_ids)
        z = rng.standard_normal((self.n_simulations, len(uniques)))
        eps = rng.standard_normal((self.n_simulations, len(cust_codes)))
        return ndtr(np.sqrt(self.rho) * z[:, cust_codes] + np.sqrt(1.0 - self.rho) * eps)

    @staticmethod
    def days_from_uniforms(u: np.ndarray, cdf: np.ndarray) -> np.ndarray:
        """First day index d with CDF[d] >= U (CDF non-decreasing); len(cdf) means not paid.

        `u` is (n_sims,) with `cdf` (H,), or (n_sims, n) with `cdf` (n, H)."""
        if u.ndim == 1:
            return np.searchsorted(cdf, u, side="left")
        days = np.empty(u.shape, dtype=np.int64)
        for i in range(u.shape[1]):
            days[:, i] = np.searchsorted(cdf[i], u[:, i], side="left")
        return days

    def sample_payment_days(self, customer_ids: pd.Series, payment_probs: np.ndarray) -> np.ndarray:
        """Simulated arrival day index per (path, invoice); horizon_days means not paid in horizon."""
        return self.days_from_uniforms(self.sample_uniforms(customer_ids), payment_probs)

    def build_outflows(self, expenses_df: pd.DataFrame, as_of: pd.Timestamp) -> np.ndarray:
        if len(expenses_df) == 0 or "due_date" not in expenses_df.columns:
            return np.zeros(self.horizon_days, dtype=float)
        offset = (pd.to_datetime(expenses_df["due_date"]) - as_of).dt.days.to_numpy() - 1
        in_horizon = (offset >= 0) & (offset < self.horizon_days)
        return np.bincount(
            offset[in_horizon], weights=expenses_df["amount"].to_numpy(dtype=float)[in_horizon],
            minlength=self.horizon_days,
        )

    def build_inflows(self, payment_days: np.ndarray, amounts: np.ndarray) -> np.ndarray:
        """(n_sims, horizon_days) matrix of receipts; unpaid-in-horizon invoices drop out."""
        H = self.horizon_days
        flat = (np.arange(self.n_simulations)[:, None] * (H + 1) + payment_days).ravel()
        weights = np.broadcast_to(amounts, payment_days.shape).ravel()
        inflows = np.bincount(flat, weights=weights, minlength=self.n_simulations * (H + 1))
        return inflows.reshape(self.n_simulations, H + 1)[:, :H]

    @staticmethod
    def deficit_probability(trajectories: np.ndarray, days: int) -> float:
        return float((trajectories[:, :days] < 0).any(axis=1).mean())

    # ------------------------------------------------------------------ #
    # Counterfactual attribution
    # ------------------------------------------------------------------ #
    def _counterfactual_risk(
        self,
        trajectories: np.ndarray,
        payment_days: np.ndarray,
        open_invoices_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """P_base - P_cf for each invoice, where P_cf assumes invoice k is paid on its due date
        (day 1 if already due/overdue). Uses the same paths as the baseline, so only invoice
        k's arrival moves: cash_cf[t] = cash[t] + amount_k * (1[t >= cf_day] - 1[t >= base_day])."""
        H30 = COUNTERFACTUAL_HORIZON_DAYS
        base = trajectories[:, :H30]
        p_base = float((base < 0).any(axis=1).mean())
        amounts = open_invoices_df["amount"].to_numpy(dtype=float)
        due_offset = (pd.to_datetime(open_invoices_df["due_date"]) - as_of).dt.days.to_numpy() - 1
        cf_day = np.clip(due_offset, 0, None)  # due today / overdue -> paid on day 1
        t = np.arange(H30)[None, :]

        p_cf = np.empty(len(amounts))
        for k in range(len(amounts)):
            shift = amounts[k] * ((t >= cf_day[k]).astype(float) - (t >= payment_days[:, k:k + 1]))
            p_cf[k] = ((base + shift) < 0).any(axis=1).mean()

        out = open_invoices_df[["invoice_id", "customer_id", "amount", "due_date"]].copy()
        if "customer_name" in open_invoices_df:
            out["customer_name"] = open_invoices_df["customer_name"].to_numpy()
        out["counterfactual_payment_day"] = cf_day + 1
        out["p_paid_in_30d"] = (payment_days < H30).mean(axis=0)
        out["p_deficit_30d_cf"] = p_cf
        out["marginal_risk_reduction"] = np.maximum(0.0, p_base - p_cf)
        return out.sort_values("marginal_risk_reduction", ascending=False, kind="stable").reset_index(drop=True)

    def joint_counterfactual_risk(self, invoice_ids: List[str], days: int = COUNTERFACTUAL_HORIZON_DAYS) -> float:
        """P(cash < 0 within `days`) if every listed invoice is paid on its due date (day 1 if already due),
        on the same simulated paths as the last run(). Same shift as _counterfactual_risk, summed over invoices."""
        if self.trajectories_ is None:
            raise RuntimeError("call run() first")
        cash = self.trajectories_[:, :days].copy()
        t = np.arange(days)[None, :]
        for inv_id in invoice_ids:
            k = self.open_invoice_ids_.index(inv_id)
            row = self.open_invoices_.iloc[k]
            cf_day = max(0, (pd.Timestamp(row["due_date"]) - self.as_of_).days - 1)
            cash += float(row["amount"]) * ((t >= cf_day).astype(float) - (t >= self.payment_days_[:, k:k + 1]))
        return float((cash < 0).any(axis=1).mean())

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        initial_cash: float,
        open_invoices_df: pd.DataFrame,
        payment_probs: np.ndarray,
        expenses_df: pd.DataFrame,
        as_of_date: str,
        compute_risk_drivers: bool = True,
    ) -> SimulationResult:
        """Simulate cash paths from `as_of_date` and attribute 30-day deficit risk.

        Args:
            initial_cash: closing cash balance on as_of_date (Rs).
            open_invoices_df: unpaid invoices; needs invoice_id, customer_id, amount, due_date
                (customer_name optional, used for display).
            payment_probs: (n_invoices, horizon_days) conditional payment CDF, row-aligned with
                open_invoices_df (InvoiceSurvivalModel.predict_cumulative_payment_prob).
            expenses_df: obligations with due_date and amount; only those in the horizon count.
            as_of_date: evaluation date, YYYY-MM-DD.
            compute_risk_drivers: run the per-invoice counterfactuals (skip for backtests).
        """
        open_invoices_df = open_invoices_df.reset_index(drop=True)
        payment_probs = np.asarray(payment_probs, dtype=float)
        expected = (len(open_invoices_df), self.horizon_days)
        if payment_probs.shape != expected:
            raise ValueError(f"payment_probs shape {payment_probs.shape} != {expected}")
        as_of = pd.Timestamp(as_of_date)
        H = self.horizon_days

        amounts = open_invoices_df["amount"].to_numpy(dtype=float)
        uniforms = self.sample_uniforms(open_invoices_df["customer_id"])
        payment_days = self.days_from_uniforms(uniforms, payment_probs)
        inflows = self.build_inflows(payment_days, amounts)
        outflows = self.build_outflows(expenses_df, as_of)
        trajectories = initial_cash + np.cumsum(inflows - outflows[None, :], axis=1)

        # First deficit day (1-based) per path; paths that never go negative are excluded.
        negative = trajectories < 0
        went_negative = negative.any(axis=1)
        first_day = negative.argmax(axis=1)[went_negative] + 1
        dates = [(as_of + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, H + 1)]
        window_days: Optional[Tuple[int, int]] = None
        window_dates: Optional[Tuple[str, str]] = None
        shortfall_date: Optional[str] = None
        if first_day.size:
            lo, med, hi = (int(round(q)) for q in np.percentile(first_day, [10, 50, 90]))
            window_days = (lo, hi)
            window_dates = (dates[lo - 1], dates[hi - 1])
            shortfall_date = dates[med - 1]

        risk = None
        top = pd.DataFrame(columns=["marginal_risk_reduction"])
        if compute_risk_drivers and len(open_invoices_df):
            risk = self._counterfactual_risk(trajectories, payment_days, open_invoices_df, as_of)
            top = risk[risk["marginal_risk_reduction"] > 0].head(TOP_K_RISK_DRIVERS)
        top_risk_invoices: List[dict] = [
            {
                "invoice_id": r.invoice_id,
                "customer_id": r.customer_id,
                "customer": getattr(r, "customer_name", r.customer_id),
                "amount": float(r.amount),
                "due_date": pd.Timestamp(r.due_date).strftime("%Y-%m-%d"),
                "p_paid_in_30d": float(r.p_paid_in_30d),
                "marginal_risk_reduction": float(r.marginal_risk_reduction),
            }
            for r in top.itertuples(index=False)
        ]

        self.uniforms_, self.payment_probs_, self.payment_days_ = uniforms, payment_probs, payment_days
        self.trajectories_, self.outflows_, self.risk_table_ = trajectories, outflows, risk
        self.open_invoice_ids_ = open_invoices_df["invoice_id"].tolist()
        self.open_invoices_, self.as_of_ = open_invoices_df, as_of
        p10, p50, p90 = np.percentile(trajectories, [10, 50, 90], axis=0)
        return SimulationResult(
            trajectories=trajectories,
            dates=dates,
            p_deficit_14d=self.deficit_probability(trajectories, 14),
            p_deficit_30d=self.deficit_probability(trajectories, 30),
            p_deficit_60d=self.deficit_probability(trajectories, min(60, H)),
            expected_shortfall_date=shortfall_date,
            p10_trajectory=p10.tolist(),
            p50_trajectory=p50.tolist(),
            p90_trajectory=p90.tolist(),
            top_risk_invoices=top_risk_invoices,
            shortfall_window_days=window_days,
            shortfall_window_dates=window_dates,
        )


def _infer_as_of(ledger: pd.DataFrame) -> str:
    """For censored rows the generator writes delay_days = as_of - due_date."""
    open_rows = ledger[~ledger["event_observed"].astype(bool)]
    as_of = (open_rows["due_date"] + pd.to_timedelta(open_rows["delay_days"], unit="D")).mode()
    return as_of.iloc[0].strftime("%Y-%m-%d")


if __name__ == "__main__":
    import time

    from engine.survival_model import InvoiceSurvivalModel
    from data.generator import generate_msme_ledger

    data_dir = Path(__file__).resolve().parents[1] / "data"
    for name in ("sample_ledger.csv", "sample_expenses.csv"):
        if not (data_dir / name).exists():
            raise SystemExit(f"{data_dir / name} not found - run `python data/generator.py` first")
    ledger = pd.read_csv(data_dir / "sample_ledger.csv", parse_dates=["issue_date", "due_date", "actual_payment_date"])
    expenses = pd.read_csv(data_dir / "sample_expenses.csv", parse_dates=["due_date"])
    as_of = _infer_as_of(ledger)
    # The CSV doesn't store the starting cash; regenerate it with the same default seed.
    _, _, initial_cash = generate_msme_ledger()

    model = InvoiceSurvivalModel().fit(ledger)
    open_inv = ledger[~ledger["event_observed"].astype(bool)].reset_index(drop=True)
    probs = model.predict_cumulative_payment_prob(open_inv, max_horizon_days=60)

    sim = CashFlowSimulator(n_simulations=5000, horizon_days=60, rho=0.5, seed=42)
    t0 = time.perf_counter()
    result = sim.run(initial_cash, open_inv, probs, expenses, as_of)
    elapsed = time.perf_counter() - t0

    lakh = lambda x: f"Rs {x / 1e5:>8,.2f} L"  # noqa: E731
    print("=" * 72)
    print(f"Runway Monte Carlo | as-of {as_of} | {sim.n_simulations:,} paths x {sim.horizon_days} days "
          f"| {len(open_inv)} open invoices | {elapsed:.2f}s")
    print("=" * 72)
    print(f"Starting cash          : {lakh(initial_cash)}")
    print(f"Obligations in horizon : {lakh(sim.outflows_.sum())}")
    print(f"Open receivables       : {lakh(open_inv['amount'].sum())}")
    print(f"\nP(deficit) within 14d  : {result.p_deficit_14d:6.1%}")
    print(f"P(deficit) within 30d  : {result.p_deficit_30d:6.1%}")
    print(f"P(deficit) within 60d  : {result.p_deficit_60d:6.1%}")
    if result.shortfall_window_days:
        lo, hi = result.shortfall_window_days
        print(f"Shortfall window       : Day {lo} to Day {hi} "
              f"({result.shortfall_window_dates[0]} -> {result.shortfall_window_dates[1]}), "
              f"median {result.expected_shortfall_date}")

    print("\nCash balance percentiles:")
    for day in (30, 60):
        print(f"  Day {day}: P10 {lakh(result.p10_trajectory[day - 1])} | "
              f"P50 {lakh(result.p50_trajectory[day - 1])} | P90 {lakh(result.p90_trajectory[day - 1])}")

    print(f"\nTop-{TOP_K_RISK_DRIVERS} risk-driver invoices (30-day deficit risk removed if paid on due date):")
    print(f"  {'invoice':<15}{'customer':<34}{'amount':>14}{'due':>12}{'P(paid 30d)':>13}{'risk cut':>10}")
    for r in result.top_risk_invoices:
        print(f"  {r['invoice_id']:<15}{r['customer']:<34}{lakh(r['amount']):>14}{r['due_date']:>12}"
              f"{r['p_paid_in_30d']:>13.0%}{r['marginal_risk_reduction']:>10.1%}")

    # --- self-test assertions ---
    T = result.trajectories
    assert T.shape == (5000, 60)
    assert 0 <= result.p_deficit_14d <= result.p_deficit_30d <= result.p_deficit_60d <= 1
    assert np.all(np.diff([result.p10_trajectory, result.p50_trajectory, result.p90_trajectory], axis=0) >= 0)
    # Cash conservation: every path's net change = receipts collected - obligations paid.
    received = sim.build_inflows(sim.payment_days_, open_inv["amount"].to_numpy()).sum(axis=1)
    assert np.allclose(T[:, -1], initial_cash + received - sim.outflows_.sum())
    # Copula preserves each invoice's marginal: simulated P(paid by day 30) ~ model CDF.
    assert np.abs((sim.payment_days_ < 30).mean(axis=0) - probs[:, 29]).max() < 0.03
    # Same seed -> identical paths (common random numbers for what-if comparisons).
    again = CashFlowSimulator(seed=42).run(initial_cash, open_inv, probs, expenses, as_of)
    assert np.array_equal(again.trajectories, T)
    # Correlation raises tail risk: rho = 0 should not give fatter deficit tails than rho = 0.5 at P10.
    indep = CashFlowSimulator(rho=0.0, seed=42).run(initial_cash, open_inv, probs, expenses, as_of)
    print(f"\nrho=0.0 vs rho=0.5 -> P(deficit 30d) {indep.p_deficit_30d:.1%} vs {result.p_deficit_30d:.1%}; "
          f"Day-30 P10 {lakh(indep.p10_trajectory[29])} vs {lakh(result.p10_trajectory[29])}")
    print("All self-test assertions passed.")
