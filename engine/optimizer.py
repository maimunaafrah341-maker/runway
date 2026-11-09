"""
Intervention optimizer for Runway ("Act").

For each high-risk open invoice, every applicable action is applied to that invoice's
payment-timing CDF, the Monte Carlo paths are re-evaluated, and actions are ranked by

    net_benefit = E[shortfall avoided] - cost
    E[shortfall] = mean over paths of max(0, -min cash over the next 30 days)

Re-evaluation uses common random numbers: the intervened invoice is re-sampled from
its *new* CDF with the *same* copula uniform U it had in the baseline, and every other
invoice keeps its baseline arrival day. This is exactly what a full re-run of
CashFlowSimulator with the modified CDF and the same seed produces (verified in the
self-test), but costs one (n_sims x 30) update per action instead of a full run.

Action effect models (d = day index after the as-of date, 0 = tomorrow):
    DIRECT_CHASE        CDF shifted 10 days earlier: F'(d) = F(d + 10).
    EARLY_DISCOUNT_2%   75% pay within 7 days (uniformly over days 1-7) and receive 98%;
                        the rest follow the original CDF. Not offered on disputed invoices.
    SPLIT_PAYMENT       With probability SPLIT_ACCEPT_PROB the customer pays 50% within
                        3 days and 50% on day 30; otherwise the original CDF.
                        Not offered on disputed invoices.
    LEGAL_MSMED_NOTICE  Only for disputed or > 45-day overdue invoices: 50% settle within
                        14 days (uniformly); the rest follow the original CDF.
Every modified CDF is floored at the original one (an action never makes payment later).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from data.schemas import ActionRecommendation
    from engine.monte_carlo import CashFlowSimulator
except ImportError:  # executed as `python engine/optimizer.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.schemas import ActionRecommendation
    from engine.monte_carlo import CashFlowSimulator

EVAL_HORIZON_DAYS = 30

CHASE_SHIFT_DAYS = 10
CHASE_COST = 500.0
DISCOUNT_RATE = 0.02
DISCOUNT_WINDOW_DAYS = 7
DISCOUNT_UPTAKE = 0.75
SPLIT_ACCEPT_PROB = 0.60          # assumption: not specified in the brief
SPLIT_FIRST_TRANCHE_DAYS = 3
SPLIT_SECOND_TRANCHE_DAY = 30
SPLIT_COST = 300.0
LEGAL_OVERDUE_THRESHOLD_DAYS = 45
LEGAL_SETTLE_PROB = 0.50
LEGAL_SETTLE_WINDOW_DAYS = 14
LEGAL_COST = 2_500.0

# An action is a list of tranches: (share of invoice amount received, CDF of arrival day).
Tranches = List[Tuple[float, np.ndarray]]


def _ramp(H: int, window: int) -> np.ndarray:
    """CDF of a payment uniformly spread over days 1..window."""
    return np.minimum((np.arange(H) + 1) / window, 1.0)


def _step(H: int, day: int) -> np.ndarray:
    """CDF of a payment arriving exactly on `day` (1-based)."""
    return (np.arange(H) >= day - 1).astype(float)


def _mix(cdf: np.ndarray, p: float, action_cdf: np.ndarray) -> np.ndarray:
    return np.maximum(cdf, p * action_cdf + (1.0 - p) * cdf)


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    eligible: Callable[[pd.Series, int], bool]          # (invoice row, days overdue) -> bool
    tranches: Callable[[np.ndarray], Tranches]          # original CDF -> tranches
    cost: Callable[[float], float]                      # invoice amount -> Rs cost
    describe: Callable[[pd.Series, int], str]


def _name(row: pd.Series) -> str:
    return row.get("customer_name", row["customer_id"])


def _lakh(x: float) -> str:
    return f"Rs {x / 1e5:,.2f} L"


def _overdue_text(days_overdue: int) -> str:
    return f"{days_overdue} days overdue" if days_overdue > 0 else f"due in {-days_overdue} days"


def _chase(cdf: np.ndarray) -> Tranches:
    padded = np.concatenate([cdf, np.full(CHASE_SHIFT_DAYS, cdf[-1])])
    return [(1.0, np.maximum(cdf, padded[CHASE_SHIFT_DAYS:]))]


ACTIONS: Tuple[ActionSpec, ...] = (
    ActionSpec(
        "DIRECT_CHASE",
        eligible=lambda row, od: True,
        tranches=_chase,
        cost=lambda amt: CHASE_COST,
        describe=lambda row, od: f"Call / WhatsApp {_name(row)}'s accounts head today about "
                                 f"{row['invoice_id']} ({_lakh(row['amount'])}, {_overdue_text(od)}).",
    ),
    ActionSpec(
        "EARLY_DISCOUNT_2%",
        eligible=lambda row, od: not bool(row["is_disputed"]),
        tranches=lambda cdf: [(1.0 - DISCOUNT_RATE,
                               _mix(cdf, DISCOUNT_UPTAKE, _ramp(len(cdf), DISCOUNT_WINDOW_DAYS)))],
        cost=lambda amt: DISCOUNT_RATE * amt,
        describe=lambda row, od: f"Offer {_name(row)} a 2% cash discount on {row['invoice_id']} "
                                 f"if paid within {DISCOUNT_WINDOW_DAYS} days "
                                 f"(you receive {_lakh(row['amount'] * (1 - DISCOUNT_RATE))}).",
    ),
    ActionSpec(
        "SPLIT_PAYMENT",
        eligible=lambda row, od: not bool(row["is_disputed"]),
        tranches=lambda cdf: [
            (0.5, _mix(cdf, SPLIT_ACCEPT_PROB, _ramp(len(cdf), SPLIT_FIRST_TRANCHE_DAYS))),
            (0.5, _mix(cdf, SPLIT_ACCEPT_PROB, _step(len(cdf), SPLIT_SECOND_TRANCHE_DAY))),
        ],
        cost=lambda amt: SPLIT_COST,
        describe=lambda row, od: f"Propose to {_name(row)}: 50% of {row['invoice_id']} now "
                                 f"({_lakh(row['amount'] / 2)}), balance in 30 days.",
    ),
    ActionSpec(
        "LEGAL_MSMED_NOTICE",
        eligible=lambda row, od: bool(row["is_disputed"]) or od > LEGAL_OVERDUE_THRESHOLD_DAYS,
        tranches=lambda cdf: [(1.0, _mix(cdf, LEGAL_SETTLE_PROB, _ramp(len(cdf), LEGAL_SETTLE_WINDOW_DAYS)))],
        cost=lambda amt: LEGAL_COST,
        describe=lambda row, od: f"Send {_name(row)} a formal MSMED Act 2006 payment claim for "
                                 f"{row['invoice_id']} ({_lakh(row['amount'])}, {_overdue_text(od)}); "
                                 f"interest at 3x the RBI bank rate accrues on delays past 45 days.",
    ),
)
ACTION_TYPES = tuple(a.action_type for a in ACTIONS)


def expected_shortfall(trajectories: np.ndarray, days: int = EVAL_HORIZON_DAYS) -> float:
    """Mean over paths of max(0, -min cash over the first `days` days), in Rs."""
    return float(np.maximum(0.0, -trajectories[:, :days].min(axis=1)).mean())


class InterventionOptimizer:
    """Ranks interventions on high-risk invoices by expected net Rs benefit."""

    def __init__(self, eval_horizon_days: int = EVAL_HORIZON_DAYS) -> None:
        self.eval_horizon_days = eval_horizon_days
        self.evaluations_: pd.DataFrame | None = None  # every (invoice, action) scored, incl. ineligible

    def _intervened_paths(
        self, base: np.ndarray, u_k: np.ndarray, base_day_k: np.ndarray, amount: float, tranches: Tranches
    ) -> np.ndarray:
        t = np.arange(base.shape[1])[None, :]
        delta = -amount * (t >= base_day_k[:, None])
        for share, cdf in tranches:
            new_day = CashFlowSimulator.days_from_uniforms(u_k, cdf)
            delta = delta + share * amount * (t >= new_day[:, None])
        return base + delta

    def recommend_actions(
        self,
        simulator: CashFlowSimulator,
        open_invoices_df: pd.DataFrame,
        top_risk_invoices: Sequence[dict],
        expenses_df: pd.DataFrame,
        initial_cash: float,
        as_of_date: str,
        best_per_invoice: bool = False,
    ) -> List[ActionRecommendation]:
        """Returns actions ranked strictly by net_benefit descending.

        `simulator` must already have been `run()` on `open_invoices_df` (same rows, same
        order); its copula draws and baseline arrival days are reused. Set
        `best_per_invoice=True` to keep only the top action for each invoice.
        """
        open_invoices_df = open_invoices_df.reset_index(drop=True)
        if simulator.uniforms_ is None or simulator.open_invoice_ids_ != open_invoices_df["invoice_id"].tolist():
            raise ValueError("Run simulator.run(...) on the same open_invoices_df before recommending actions")
        H = min(self.eval_horizon_days, simulator.horizon_days)
        as_of = pd.Timestamp(as_of_date)

        # Baseline paths over the evaluation window, rebuilt from the stored arrival days.
        amounts = open_invoices_df["amount"].to_numpy(dtype=float)
        inflows = simulator.build_inflows(simulator.payment_days_, amounts)
        outflows = simulator.build_outflows(expenses_df, as_of)
        base = (initial_cash + np.cumsum(inflows - outflows[None, :], axis=1))[:, :H]
        base_deficit = np.maximum(0.0, -base.min(axis=1))
        base_es = float(base_deficit.mean())
        at_risk = base_deficit > 0

        row_of: Dict[str, int] = {inv: i for i, inv in enumerate(open_invoices_df["invoice_id"])}
        records, evaluations = [], []
        for item in top_risk_invoices:
            k = row_of.get(item["invoice_id"])
            if k is None:
                raise KeyError(f"{item['invoice_id']} is not in open_invoices_df")
            row = open_invoices_df.iloc[k]
            days_overdue = int((as_of - pd.Timestamp(row["due_date"])).days)
            for spec in ACTIONS:
                eligible = spec.eligible(row, days_overdue)
                evaluation = {"invoice_id": row["invoice_id"], "action_type": spec.action_type, "eligible": eligible}
                evaluations.append(evaluation)
                if not eligible:
                    continue
                paths = self._intervened_paths(
                    base, simulator.uniforms_[:, k], simulator.payment_days_[:, k],
                    float(row["amount"]), spec.tranches(simulator.payment_probs_[k]),
                )
                post_deficit = np.maximum(0.0, -paths.min(axis=1))
                avoided = base_es - float(post_deficit.mean())
                cost = float(spec.cost(float(row["amount"])))
                # Share of at-risk paths in which this action actually shrinks the shortfall.
                confidence = float((post_deficit[at_risk] < base_deficit[at_risk] - 1e-6).mean()) if at_risk.any() else 0.0
                evaluation.update(shortfall_avoided=avoided, cost=cost, net_benefit=avoided - cost,
                                  p_deficit_30d_after=float((post_deficit > 0).mean()))
                records.append(ActionRecommendation(
                    invoice_id=row["invoice_id"],
                    customer_id=row["customer_id"],
                    action_type=spec.action_type,
                    description=spec.describe(row, days_overdue),
                    expected_shortfall_avoided=round(avoided, 2),
                    cost_of_action=round(cost, 2),
                    net_benefit=round(avoided - cost, 2),
                    confidence_score=round(confidence, 4),
                ))

        self.baseline_expected_shortfall_ = base_es
        self.baseline_p_deficit_ = float(at_risk.mean())
        self.evaluations_ = pd.DataFrame(evaluations)
        records.sort(key=lambda r: r.net_benefit, reverse=True)
        if best_per_invoice:
            seen, best = set(), []
            for r in records:
                if r.invoice_id not in seen:
                    seen.add(r.invoice_id)
                    best.append(r)
            records = best
        return records


if __name__ == "__main__":
    from data.generator import generate_msme_ledger
    from engine.survival_model import InvoiceSurvivalModel

    invoices, expenses, cash = generate_msme_ledger()
    as_of = invoices.attrs["as_of_date"]
    open_inv = invoices[~invoices["event_observed"]].reset_index(drop=True)
    probs = InvoiceSurvivalModel().fit(invoices).predict_cumulative_payment_prob(open_inv, 60)

    sim = CashFlowSimulator(n_simulations=5000, horizon_days=60, rho=0.5, seed=42)
    result = sim.run(cash, open_inv, probs, expenses, as_of)

    opt = InterventionOptimizer()
    recs = opt.recommend_actions(sim, open_inv, result.top_risk_invoices, expenses, cash, as_of)

    print("=" * 100)
    print(f"Baseline 30d: P(deficit) {opt.baseline_p_deficit_:.1%} | expected shortfall {_lakh(opt.baseline_expected_shortfall_)}")
    print("=" * 100)
    print(f"{'#':>2} {'action':<20}{'invoice':<15}{'avoided':>14}{'cost':>12}{'net benefit':>14}{'conf':>7}")
    for i, r in enumerate(recs, 1):
        print(f"{i:>2} {r.action_type:<20}{r.invoice_id:<15}{_lakh(r.expected_shortfall_avoided):>14}"
              f"{'Rs ' + format(r.cost_of_action, ',.0f'):>12}{_lakh(r.net_benefit):>14}{r.confidence_score:>7.0%}")
    print("\nTop recommendation:", recs[0].description)
    legal = [r for r in recs if r.action_type == "LEGAL_MSMED_NOTICE"]
    if legal:
        print("Legal route example:", legal[0].description)

    # --- self-test assertions ---
    nets = [r.net_benefit for r in recs]
    assert nets == sorted(nets, reverse=True), "not ranked by net benefit"
    assert all(abs(r.net_benefit - (r.expected_shortfall_avoided - r.cost_of_action)) < 0.02 for r in recs)
    assert all(r.action_type in ACTION_TYPES for r in recs)
    ev = opt.evaluations_
    assert not ev[(ev.action_type == "LEGAL_MSMED_NOTICE") & ev.eligible].empty or \
        not any(r["invoice_id"] in set(open_inv.loc[open_inv.is_disputed, "invoice_id"]) for r in result.top_risk_invoices)
    for r in recs:
        if r.action_type == "EARLY_DISCOUNT_2%":
            amt = float(open_inv.loc[open_inv.invoice_id == r.invoice_id, "amount"].iloc[0])
            assert abs(r.cost_of_action - 0.02 * amt) < 0.01
    # Fast path == full Monte Carlo re-run with the modified CDF and the same seed.
    target = recs[[r.action_type for r in recs].index("DIRECT_CHASE")]
    k = open_inv.index[open_inv.invoice_id == target.invoice_id][0]
    probs_cf = probs.copy()
    probs_cf[k] = _chase(probs[k])[0][1]
    rerun = CashFlowSimulator(seed=42).run(cash, open_inv, probs_cf, expenses, as_of, compute_risk_drivers=False)
    full_avoided = opt.baseline_expected_shortfall_ - expected_shortfall(rerun.trajectories)
    assert abs(full_avoided - target.expected_shortfall_avoided) < 1.0, (full_avoided, target.expected_shortfall_avoided)
    print(f"\nFast path matches full re-run for {target.invoice_id} DIRECT_CHASE ({_lakh(full_avoided)} avoided).")
    print("All self-test assertions passed.")
