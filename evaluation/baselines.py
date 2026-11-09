"""
The three required baselines for Runway's 30-day deficit forecast.

Each baseline takes the point-in-time state at evaluation date t and returns
P(cash < 0 on any of days t+1 .. t+window):

1. NaiveDueDateBaseline     every open invoice is paid exactly on its due date
                            (on day 1 if already overdue). Deterministic -> P in {0, 1}.
2. CustomerMeanDelayBaseline due date shifted by the customer's historical mean delay
                            (segment / global mean for customers with no history).
                            Deterministic -> P in {0, 1}.
3. ArimaBaseline            ARIMA(1, 1, 0) on the daily net cash flow series; P is the
                            share of simulated future paths whose cash dips below zero.
                            Purely statistical: it does not see invoices or the expense calendar.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _deterministic_deficit(
    cash_t: float, arrival_offsets: np.ndarray, amounts: np.ndarray, outflows: np.ndarray
) -> float:
    """1.0 if the single forecast path dips below zero. arrival_offsets are 1-based days after t."""
    window = len(outflows)
    inflows = np.zeros(window)
    in_window = (arrival_offsets >= 1) & (arrival_offsets <= window)
    np.add.at(inflows, arrival_offsets[in_window] - 1, amounts[in_window])
    path = cash_t + np.cumsum(inflows - outflows)
    return float((path < 0).any())


class NaiveDueDateBaseline:
    name = "Naive Due-Date"

    def predict(self, cash_t: float, t: pd.Timestamp, ledger_t: pd.DataFrame, open_t: pd.DataFrame,
                outflows: np.ndarray, net_flow_history: pd.Series) -> float:
        offsets = np.maximum((open_t["due_date"] - t).dt.days.to_numpy(), 1)
        return _deterministic_deficit(cash_t, offsets, open_t["amount"].to_numpy(float), outflows)


class CustomerMeanDelayBaseline:
    name = "Customer Mean Delay"

    def predict(self, cash_t: float, t: pd.Timestamp, ledger_t: pd.DataFrame, open_t: pd.DataFrame,
                outflows: np.ndarray, net_flow_history: pd.Series) -> float:
        paid = ledger_t[ledger_t["event_observed"]]
        by_customer = paid.groupby("customer_id")["delay_days"].mean()
        by_segment = paid.groupby("customer_segment")["delay_days"].mean()
        overall = float(paid["delay_days"].mean()) if len(paid) else 0.0
        delay = (open_t["customer_id"].map(by_customer)
                 .fillna(open_t["customer_segment"].map(by_segment))
                 .fillna(overall))
        expected = open_t["due_date"] + pd.to_timedelta(delay.round(), unit="D")
        offsets = np.maximum((expected - t).dt.days.to_numpy(), 1)
        return _deterministic_deficit(cash_t, offsets, open_t["amount"].to_numpy(float), outflows)


class ArimaBaseline:
    name = "ARIMA(1,1,0)"

    def __init__(self, order=(1, 1, 0), n_paths: int = 2000, seed: int = 42) -> None:
        self.order = order
        self.n_paths = n_paths
        self.seed = seed

    def predict(self, cash_t: float, t: pd.Timestamp, ledger_t: pd.DataFrame, open_t: pd.DataFrame,
                outflows: np.ndarray, net_flow_history: pd.Series) -> float:
        from statsmodels.tsa.arima.model import ARIMA

        y = net_flow_history.to_numpy(dtype=float) / 1e5  # fit in lakhs for numerical stability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y, order=self.order).fit()
            sims = res.simulate(nsimulations=len(outflows), repetitions=self.n_paths, anchor="end",
                                rng=np.random.default_rng(self.seed))
        flows = np.asarray(sims).reshape(len(outflows), self.n_paths).T * 1e5  # (paths, window)
        paths = cash_t + np.cumsum(flows, axis=1)
        return float((paths < 0).any(axis=1).mean())


BASELINES = (NaiveDueDateBaseline, CustomerMeanDelayBaseline, ArimaBaseline)
