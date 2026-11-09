"""
Rolling backtest of Runway's 30-day deficit probability against the three baselines.

Evaluation points t run daily (by default) over the 90 days that end `window` days
before the as-of date, so every point has a fully realised 30-day outcome:

    t in [as_of - window - lookback + 1, as_of - window]

At each t every model sees only point-in-time data (no look-ahead):
  * invoices issued on or before t; payments after t are hidden (censored at t)
  * the survival model is re-fitted on that censored ledger
  * starting cash is the actual reconstructed closing balance on t
  * outflows are the scheduled obligations due in (t, t + window]

Outcome:  Y_t = 1 if actual closing cash < 0 on any day in t+1 .. t+window.

Metrics (per model):
  * 30d Brier score             mean((P_t - Y_t)^2), lower is better
  * Warning lead time (days)    for each deficit onset D (cash goes from >= 0 to < 0) inside
                                the evaluated range: D - (earliest t in [D - window, D - 1]
                                with P_t > threshold); 0 if never warned. Averaged over onsets.
  * False warning rate (%)      share of warnings (P_t > threshold) with Y_t = 0
  * Onsets caught               onsets with at least one warning in the preceding window
Note: daily evaluation points overlap heavily, so they are far from independent samples.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from data.generator import reconstruct_cash_history
    from engine.monte_carlo import CashFlowSimulator
    from engine.survival_model import InvoiceSurvivalModel
    from evaluation.baselines import BASELINES
except ImportError:  # executed as `python evaluation/backtest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.generator import reconstruct_cash_history
    from engine.monte_carlo import CashFlowSimulator
    from engine.survival_model import InvoiceSurvivalModel
    from evaluation.baselines import BASELINES

RUNWAY_NAME = "Runway (Survival + Monte Carlo)"
WARNING_THRESHOLD = 0.40


def ledger_as_of(invoices_df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """The invoice ledger as it looked at close of day t."""
    df = invoices_df[invoices_df["issue_date"] <= t].copy()
    df.loc[df["actual_payment_date"] > t, "actual_payment_date"] = pd.NaT
    df["event_observed"] = df["actual_payment_date"].notna()
    df["delay_days"] = (df["actual_payment_date"].fillna(t) - df["due_date"]).dt.days.astype(int)
    return df.reset_index(drop=True)


class RunwayPredictor:
    name = RUNWAY_NAME

    def __init__(self, n_simulations: int = 2000, rho: float = 0.5, seed: int = 42) -> None:
        self.n_simulations, self.rho, self.seed = n_simulations, rho, seed

    def predict(self, cash_t: float, t: pd.Timestamp, ledger_t: pd.DataFrame, open_t: pd.DataFrame,
                outflows: np.ndarray, net_flow_history: pd.Series, expenses_df: pd.DataFrame) -> float:
        window = len(outflows)
        model = InvoiceSurvivalModel().fit(ledger_t)
        probs = model.predict_cumulative_payment_prob(open_t, max_horizon_days=window)
        sim = CashFlowSimulator(n_simulations=self.n_simulations, horizon_days=window, rho=self.rho, seed=self.seed)
        result = sim.run(cash_t, open_t, probs, expenses_df, t.strftime("%Y-%m-%d"), compute_risk_drivers=False)
        return result.p_deficit_30d


class RollingBacktest:
    def __init__(
        self,
        lookback_days: int = 90,
        window_days: int = 30,
        step_days: int = 1,
        warning_threshold: float = WARNING_THRESHOLD,
        n_simulations: int = 2000,
        seed: int = 42,
    ) -> None:
        if window_days != 30:
            raise ValueError("Runway's simulator reports P(deficit) at 30 days; keep window_days=30")
        self.lookback_days, self.window_days, self.step_days = lookback_days, window_days, step_days
        self.warning_threshold = warning_threshold
        self.runway = RunwayPredictor(n_simulations=n_simulations, seed=seed)
        self.baselines = [cls() for cls in BASELINES]
        self.predictions_: Optional[pd.DataFrame] = None
        self.cash_history_: Optional[pd.DataFrame] = None

    @property
    def model_names(self) -> List[str]:
        return [self.runway.name] + [b.name for b in self.baselines]

    def run(self, invoices_df: pd.DataFrame, expenses_df: pd.DataFrame, cash_as_of: float,
            verbose: bool = False) -> pd.DataFrame:
        """Returns the summary table; per-point predictions are kept in `predictions_`."""
        as_of = pd.Timestamp(invoices_df.attrs["as_of_date"])
        history = reconstruct_cash_history(invoices_df, expenses_df, cash_as_of)
        cash = history["closing_cash"]
        W = self.window_days
        last_t = as_of - pd.Timedelta(days=W)
        first_t = last_t - pd.Timedelta(days=self.lookback_days - 1)
        if first_t <= history.index[0] + pd.Timedelta(days=30):
            raise ValueError("Not enough cash history before the first evaluation point")
        exp_dates = pd.to_datetime(expenses_df["due_date"])

        rows = []
        for t in pd.date_range(first_t, last_t, freq=f"{self.step_days}D"):
            started = time.perf_counter()
            ledger_t = ledger_as_of(invoices_df, t)
            open_t = ledger_t[~ledger_t["event_observed"]].reset_index(drop=True)
            offsets = (exp_dates - t).dt.days.to_numpy()
            in_window = (offsets >= 1) & (offsets <= W)
            outflows = np.bincount(offsets[in_window] - 1, weights=expenses_df["amount"].to_numpy(float)[in_window],
                                   minlength=W)
            cash_t = float(cash.loc[t])
            net_hist = history.loc[:t, "net_flow"]
            future = cash.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=W)]
            neg = future.to_numpy() < 0

            row = {"t": t, "cash_t": cash_t, "y": int(neg.any()),
                   "first_deficit_day": int(neg.argmax()) + 1 if neg.any() else np.nan}
            row[self.runway.name] = self.runway.predict(cash_t, t, ledger_t, open_t, outflows, net_hist, expenses_df)
            for b in self.baselines:
                row[b.name] = b.predict(cash_t, t, ledger_t, open_t, outflows, net_hist)
            rows.append(row)
            if verbose:
                print(f"  {t:%Y-%m-%d} cash {cash_t / 1e5:7.2f} L  Y={row['y']}  Runway P={row[self.runway.name]:.2f}"
                      f"  ({time.perf_counter() - started:.1f}s)")

        self.predictions_ = pd.DataFrame(rows).set_index("t")
        self.cash_history_ = history
        return self.summarize()

    def summarize(self) -> pd.DataFrame:
        pred = self.predictions_
        cash = self.cash_history_["closing_cash"]
        W, thr = self.window_days, self.warning_threshold
        first_t, last_t = pred.index[0], pred.index[-1]

        is_neg = cash < 0
        onsets = cash.index[is_neg & ~is_neg.shift(1, fill_value=True)]
        onsets = [d for d in onsets if first_t < d <= last_t + pd.Timedelta(days=W)]

        table = []
        for name in self.model_names:
            p, y = pred[name], pred["y"]
            warned = p > thr
            leads, caught = [], 0
            for d in onsets:
                lookback = pred.index[(pred.index >= d - pd.Timedelta(days=W)) & (pred.index < d)]
                hits = lookback[warned.loc[lookback].to_numpy()]
                leads.append((d - hits.min()).days if len(hits) else 0)
                caught += bool(len(hits))
            table.append({
                "Model": name,
                "30d Brier Score": float(((p - y) ** 2).mean()),
                "Warning Lead Time (days)": float(np.mean(leads)) if leads else np.nan,
                "False Warning Rate (%)": float(100 * (y[warned] == 0).mean()) if warned.any() else np.nan,
                "Onsets Caught": f"{caught}/{len(onsets)}",
            })
        return pd.DataFrame(table)

    def reliability_curve(self, model: str = RUNWAY_NAME, n_bins: int = 5) -> pd.DataFrame:
        """Mean predicted vs observed deficit frequency per probability bin (calibration plot data)."""
        pred = self.predictions_
        bins = pd.cut(pred[model], np.linspace(0, 1, n_bins + 1), include_lowest=True)
        return pred.groupby(bins, observed=True).agg(
            mean_predicted=(model, "mean"), observed_rate=("y", "mean"), n=("y", "size"))


if __name__ == "__main__":
    from data.generator import generate_msme_ledger

    invoices, expenses, cash = generate_msme_ledger()
    bt = RollingBacktest(lookback_days=90, window_days=30, step_days=1, n_simulations=2000)
    started = time.perf_counter()
    summary = bt.run(invoices, expenses, cash)
    elapsed = time.perf_counter() - started

    pred = bt.predictions_
    print("=" * 100)
    print(f"Rolling backtest: {len(pred)} evaluation points {pred.index[0]:%Y-%m-%d} -> {pred.index[-1]:%Y-%m-%d}, "
          f"30-day window, warning at P > {WARNING_THRESHOLD:.0%} ({elapsed:.0f}s)")
    print(f"Points with a realised deficit (Y=1): {int(pred['y'].sum())}/{len(pred)}")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print(summary.to_string(index=False))

    print("\nRunway reliability (calibration) curve:")
    print(bt.reliability_curve().round(3).to_string())

    # --- self-test assertions (pipeline sanity, not a guarantee that Runway wins) ---
    assert set(summary["Model"]) == set(bt.model_names) and len(summary) == 4
    for name in bt.model_names:
        assert pred[name].between(0, 1).all(), name
    assert summary["30d Brier Score"].between(0, 1).all()
    # The no-look-ahead ledger must hide every payment after t.
    probe_t = pred.index[0]
    lt = ledger_as_of(invoices, probe_t)
    assert (lt["actual_payment_date"].dropna() <= probe_t).all() and (lt["issue_date"] <= probe_t).all()

    best_brier = summary.loc[summary["30d Brier Score"].idxmin(), "Model"]
    best_lead = summary.loc[summary["Warning Lead Time (days)"].idxmax(), "Model"]
    print(f"\nLowest Brier score      : {best_brier}")
    print(f"Longest warning lead    : {best_lead}")
    print("All self-test assertions passed.")
