"""
Synthetic MSME ledger generator for Runway.

Simulates 12 months of receivables and recurring obligations for an Indian
auto-component manufacturer (~Rs 5-15 Cr annual turnover), ending at an
evaluation ("as-of") date. Invoices still unpaid at that date are right-censored
(actual_payment_date is None).

Behaviours modelled
-------------------
* Segment-specific payment behaviour:
    - Tier-1 Auto OEM: pays 15-45 days past due (secure but slow approvals).
    - Mid-tier Supplier: mostly a week or two late, some early.
    - Retail Distributor: either roughly on time or very late; small bad-debt rate.
* Persistent customer frailty: each customer has a fixed lateness multiplier.
* Correlated lateness: customers go through cash-strapped episodes (monthly
  Markov chain). During an episode a single gamma ~ LogNormal(mu, sigma) is drawn
  and every invoice of that customer falling due in the episode slips by it.
* Disputes: a disputed invoice is delayed 30-90 extra days or never paid.
* Obligations: payroll (staff on the 1st, wages on the 7th), EMI and rent on the
  5th, raw-material vendors on the 10th and 25th, GSTR-3B on the 20th, and
  advance tax on 15 Jun / Sep / Dec / Mar.

Usage:
    invoices_df, expenses_df, cash = generate_msme_ledger()
    python data/generator.py   # writes data/sample_ledger.csv and data/sample_expenses.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from data.schemas import (
        CUSTOMER_SEGMENTS,
        EXPENSE_EMI_RENT,
        EXPENSE_GST_TAX,
        EXPENSE_PAYROLL,
        EXPENSE_RAW_MATERIAL,
        SEGMENT_MID_TIER,
        SEGMENT_RETAIL,
        SEGMENT_TIER1_OEM,
        InvoiceRecord,
        RecurringExpense,
    )
except ImportError:  # executed as `python data/generator.py`
    from schemas import (  # type: ignore[no-redef]
        CUSTOMER_SEGMENTS,
        EXPENSE_EMI_RENT,
        EXPENSE_GST_TAX,
        EXPENSE_PAYROLL,
        EXPENSE_RAW_MATERIAL,
        SEGMENT_MID_TIER,
        SEGMENT_RETAIL,
        SEGMENT_TIER1_OEM,
        InvoiceRecord,
        RecurringExpense,
    )

# --------------------------------------------------------------------------- #
# Calibration constants
# --------------------------------------------------------------------------- #
GST_RATE = 0.18
GST_FRACTION_OF_GROSS = GST_RATE / (1 + GST_RATE)  # GST share of a GST-inclusive amount
MIN_INVOICE_AMOUNT = 50_000.0
MAX_INVOICE_AMOUNT = 1_500_000.0
HISTORY_DAYS = 365
FORWARD_EXPENSE_DAYS = 90       # obligations generated beyond the as-of date (covers a 60-day horizon)
CASHBOOK_WARMUP_MONTHS = 2      # cash book opens once collections from the first invoices are flowing
MONTH_END_DISPATCH_SHARE = 0.25  # share of invoices bunched into the last 5 days of a month

# Correlated lateness: customer-level cash-stress regime.
STRESS_STAY_PROB = 0.55          # P(stressed next month | stressed this month)
STRESS_GAMMA_MU = 0.45           # gamma ~ LogNormal(mu, sigma); median ~1.57x
STRESS_GAMMA_SIGMA = 0.30
STRESS_SLIP_DAYS = 12.0          # additive slip per unit of (gamma - 1), so on-time payers also slip

DISPUTE_NEVER_PAID_SHARE = 0.25
TARGET_NET_MARGIN = 0.01         # net cash flow over the history window as a share of collections


@dataclass(frozen=True)
class SegmentProfile:
    customer_share: float          # share of the customer base
    invoice_weight: float          # relative ordering frequency per customer
    amount_median: float           # Rs, GST-inclusive
    amount_sigma: float            # log-normal dispersion
    terms_probs: Dict[int, float]  # payment terms -> probability
    dispute_prob: float
    bad_debt_prob: float           # invoice never paid even without a dispute
    stress_enter_prob: float       # P(customer enters a cash-stress episode in a given month)


SEGMENT_PROFILES: Dict[str, SegmentProfile] = {
    SEGMENT_TIER1_OEM: SegmentProfile(
        customer_share=0.16, invoice_weight=3.0, amount_median=380_000, amount_sigma=0.45,
        terms_probs={30: 0.10, 45: 0.40, 60: 0.50},
        dispute_prob=0.02, bad_debt_prob=0.0, stress_enter_prob=0.06,
    ),
    SEGMENT_MID_TIER: SegmentProfile(
        customer_share=0.40, invoice_weight=1.0, amount_median=180_000, amount_sigma=0.55,
        terms_probs={30: 0.35, 45: 0.45, 60: 0.20},
        dispute_prob=0.04, bad_debt_prob=0.005, stress_enter_prob=0.10,
    ),
    SEGMENT_RETAIL: SegmentProfile(
        customer_share=0.44, invoice_weight=0.8, amount_median=85_000, amount_sigma=0.60,
        terms_probs={30: 0.60, 45: 0.30, 60: 0.10},
        dispute_prob=0.06, bad_debt_prob=0.03, stress_enter_prob=0.14,
    ),
}

# Fictional counterparties (no real companies).
_NAME_STEMS = [
    "Sharma Steel", "Deccan Axle", "Godavari Motors", "Shakti Forgings", "Narmada Auto",
    "Kaveri Components", "Vindhya Engineering", "Sahyadri Castings", "Ganga Fasteners",
    "Konark Gears", "Aravali Tyres", "Malabar Auto Parts", "Indus Drivetrain", "Satpura Pressings",
    "Krishna Bearings", "Coromandel Motors", "Nilgiri Spares", "Ajanta Brakes", "Thar Radiators",
    "Chambal Auto Electricals", "Mahanadi Pistons", "Tapti Springs", "Hampi Precision",
    "Ellora Clutches", "Sindhu Wheels", "Brahmaputra Mobility", "Lakshmi Auto Agencies",
    "Ashoka Hydraulics", "Vijaya Filters", "Kalinga Chassis",
]
_NAME_SUFFIX = {
    SEGMENT_TIER1_OEM: "Ltd",
    SEGMENT_MID_TIER: "Pvt Ltd",
    SEGMENT_RETAIL: "Distributors",
}

INVOICE_COLUMNS = [
    "invoice_id", "customer_id", "customer_name", "customer_segment", "amount", "gst_amount",
    "issue_date", "due_date", "actual_payment_date", "payment_terms_days", "is_disputed",
    "customer_past_avg_delay", "customer_prior_paid_count", "delay_days", "event_observed",
]
EXPENSE_COLUMNS = ["expense_id", "category", "description", "amount", "due_date", "is_flexible", "is_projected"]


# --------------------------------------------------------------------------- #
# Customers and correlated stress regimes
# --------------------------------------------------------------------------- #
def _allocate_segments(n_customers: int) -> List[str]:
    if n_customers < len(CUSTOMER_SEGMENTS):
        raise ValueError(f"n_customers must be >= {len(CUSTOMER_SEGMENTS)}")
    counts = {s: max(1, round(SEGMENT_PROFILES[s].customer_share * n_customers)) for s in CUSTOMER_SEGMENTS}
    # Absorb rounding error in the largest segment.
    counts[SEGMENT_RETAIL] += n_customers - sum(counts.values())
    return [s for s in CUSTOMER_SEGMENTS for _ in range(counts[s])]


def _build_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    segments = _allocate_segments(n_customers)
    stems = [_NAME_STEMS[i % len(_NAME_STEMS)] for i in rng.permutation(max(n_customers, len(_NAME_STEMS)))]
    rows = []
    for i, seg in enumerate(segments):
        prof = SEGMENT_PROFILES[seg]
        stem = stems[i] if n_customers <= len(_NAME_STEMS) else f"{stems[i]} {i // len(_NAME_STEMS) + 1}"
        rows.append({
            "customer_id": f"CUST-{i + 1:03d}",
            "customer_name": f"{stem} {_NAME_SUFFIX[seg]}",
            "customer_segment": seg,
            # Persistent heterogeneity within a segment (frailty).
            "frailty": float(rng.lognormal(0.0, 0.30)),
            # Ordering intensity: Gamma noise around the segment weight.
            "order_weight": float(prof.invoice_weight * rng.gamma(4.0, 0.25)),
        })
    return pd.DataFrame(rows)


def _simulate_stress_gammas(customers: pd.DataFrame, n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n_customers, n_months) matrix of lateness multipliers; 1.0 means not stressed.

    One gamma is drawn per stress *episode*, so every invoice of a customer falling
    due within the same episode shares the same shock (correlated lateness).
    """
    gammas = np.ones((len(customers), n_months))
    for c, seg in enumerate(customers["customer_segment"]):
        p_enter = SEGMENT_PROFILES[seg].stress_enter_prob
        stressed, gamma = False, 1.0
        for m in range(n_months):
            stressed = rng.random() < (STRESS_STAY_PROB if stressed else p_enter)
            if stressed:
                if gamma == 1.0:  # new episode
                    gamma = float(rng.lognormal(STRESS_GAMMA_MU, STRESS_GAMMA_SIGMA))
                gammas[c, m] = gamma
            else:
                gamma = 1.0
    return gammas


# --------------------------------------------------------------------------- #
# Invoices
# --------------------------------------------------------------------------- #
def _sample_base_delay(segment: str, rng: np.random.Generator) -> float:
    """Days paid past due date under normal conditions (negative = early)."""
    if segment == SEGMENT_TIER1_OEM:
        if rng.random() < 0.10:
            return rng.uniform(0, 10)          # occasional clean, on-time cycle
        return rng.uniform(15, 45)             # vendor-portal / GRN approval lag
    if segment == SEGMENT_MID_TIER:
        return rng.gamma(2.0, 5.0) - 3.0       # mean ~7 days late, some early
    # Retail distributor: bimodal
    if rng.random() < 0.5:
        return rng.normal(2.0, 4.0)
    return rng.gamma(1.5, 14.0)                # mean ~21 days late, long tail


def _sample_issue_dates(n: int, start: pd.Timestamp, as_of: pd.Timestamp, rng: np.random.Generator) -> pd.DatetimeIndex:
    span = (as_of - start).days
    offsets = rng.integers(0, span, size=n)
    dates = start + pd.to_timedelta(offsets, unit="D")
    # Manufacturers push dispatches at month-end to hit monthly targets.
    bunch = rng.random(n) < MONTH_END_DISPATCH_SHARE
    month_end = dates + pd.offsets.MonthEnd(0)
    bunched = month_end - pd.to_timedelta(rng.integers(0, 5, size=n), unit="D")
    dates = dates.where(~bunch, bunched)
    last_issue = as_of - pd.Timedelta(days=1)
    return pd.DatetimeIndex(np.minimum(dates.values, last_issue.to_datetime64()))


def _past_avg_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Mean realised delay of each customer's invoices paid strictly before this issue date (no leakage)."""
    avg = np.full(len(df), np.nan)
    count = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("customer_id"):
        paid = g[g["actual_payment_date"].notna()]
        pay_dates = paid["actual_payment_date"].values
        delays = paid["delay_days"].values
        for idx, issue in zip(g.index, g["issue_date"].values):
            mask = pay_dates < issue
            count[idx] = int(mask.sum())
            if count[idx]:
                avg[idx] = float(delays[mask].mean())
    df["customer_past_avg_delay"] = np.round(avg, 1)
    df["customer_prior_paid_count"] = count
    return df


def _generate_invoices(
    customers: pd.DataFrame,
    n_invoices: int,
    start: pd.Timestamp,
    as_of: pd.Timestamp,
    rng: np.random.Generator,
) -> pd.DataFrame:
    weights = customers["order_weight"].to_numpy()
    cust_idx = rng.choice(len(customers), size=n_invoices, p=weights / weights.sum())
    issue_dates = _sample_issue_dates(n_invoices, start, as_of, rng)

    n_months = (as_of.year - start.year) * 12 + (as_of.month - start.month) + 8  # room for late payments
    gammas = _simulate_stress_gammas(customers, n_months, rng)

    rows = []
    for c, issue in zip(cust_idx, issue_dates):
        cust = customers.iloc[c]
        seg = cust["customer_segment"]
        prof = SEGMENT_PROFILES[seg]

        amount = float(np.clip(rng.lognormal(np.log(prof.amount_median), prof.amount_sigma),
                               MIN_INVOICE_AMOUNT, MAX_INVOICE_AMOUNT))
        amount = round(amount, -2)
        terms = int(rng.choice(list(prof.terms_probs), p=list(prof.terms_probs.values())))
        due = issue + pd.Timedelta(days=terms)

        base = _sample_base_delay(seg, rng)
        if base > 0:
            base *= cust["frailty"]
        month_idx = min((due.year - start.year) * 12 + (due.month - start.month), n_months - 1)
        gamma = gammas[c, month_idx]
        delay = base * gamma + (gamma - 1.0) * STRESS_SLIP_DAYS

        is_disputed = bool(rng.random() < prof.dispute_prob)
        never_paid = bool(rng.random() < prof.bad_debt_prob)
        if is_disputed:
            delay += rng.uniform(30, 90)
            never_paid = never_paid or bool(rng.random() < DISPUTE_NEVER_PAID_SHARE)

        payment: Optional[pd.Timestamp] = None
        if not never_paid:
            payment = due + pd.Timedelta(days=int(round(delay)))
            payment = max(payment, issue + pd.Timedelta(days=1))
            if payment.dayofweek == 6:  # no bank credit on Sundays
                payment += pd.Timedelta(days=1)
            if payment > as_of:         # not yet observed -> right-censored
                payment = None

        rows.append({
            "customer_id": cust["customer_id"],
            "customer_name": cust["customer_name"],
            "customer_segment": seg,
            "amount": amount,
            "gst_amount": round(amount * GST_FRACTION_OF_GROSS, 2),
            "issue_date": issue,
            "due_date": due,
            "actual_payment_date": payment,
            "payment_terms_days": terms,
            "is_disputed": is_disputed,
        })

    df = pd.DataFrame(rows)
    df["actual_payment_date"] = pd.to_datetime(df["actual_payment_date"])
    df = df.sort_values(["issue_date", "customer_id"], kind="stable").reset_index(drop=True)
    df.insert(0, "invoice_id", [f"INV-{d.year}-{i + 1:04d}" for i, d in enumerate(df["issue_date"])])

    df["event_observed"] = df["actual_payment_date"].notna()
    # Observed delay for paid invoices; days past due at the as-of date for censored ones
    # (negative if not yet due). This is the survival duration measured from the due date.
    end = df["actual_payment_date"].fillna(as_of)
    df["delay_days"] = (end - df["due_date"]).dt.days.astype(int)
    df = _past_avg_delay(df)
    return df[INVOICE_COLUMNS]


# --------------------------------------------------------------------------- #
# Recurring obligations
# --------------------------------------------------------------------------- #
def _monthly_totals(
    invoices: pd.DataFrame,
    date_col: str,
    months: pd.PeriodIndex,
    first_valid: pd.Period,
    as_of: pd.Timestamp,
) -> pd.Series:
    """Invoice value per calendar month of `date_col` (issue -> billing, payment -> collections).

    Months from as_of onwards (partial / future) use the trailing 3-month mean; months
    before `first_valid` (not yet in steady state) use the overall mean.
    """
    totals = invoices.dropna(subset=[date_col]).groupby(invoices[date_col].dt.to_period("M"))["amount"].sum()
    as_of_m = as_of.to_period("M")
    valid = [m for m in pd.period_range(first_valid, as_of_m - 1, freq="M")]
    series = totals.reindex(valid, fill_value=0.0)
    trailing = float(series.iloc[-3:].mean())
    avg = float(series.mean())
    out = {}
    for m in months:
        if m >= as_of_m:
            out[m] = trailing
        elif m < first_valid:
            out[m] = avg
        else:
            out[m] = float(series[m])
    return pd.Series(out)


def _cashbook_start(start: pd.Timestamp) -> pd.Timestamp:
    """First day of the cash book. Opening it a couple of months after the first invoice
    means early outflows are matched by collections; otherwise the ledger would have
    costs with no opening receivables to pay for them."""
    return (start.to_period("M") + CASHBOOK_WARMUP_MONTHS).start_time


def _build_expenses(
    invoices: pd.DataFrame,
    start: pd.Timestamp,
    as_of: pd.Timestamp,
    raw_material_share: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    end = as_of + pd.Timedelta(days=FORWARD_EXPENSE_DAYS)
    book_start = _cashbook_start(start)
    months = pd.period_range(start.to_period("M") - 2, end.to_period("M"), freq="M")
    rev = _monthly_totals(invoices, "issue_date", months, start.to_period("M"), as_of)
    collected = _monthly_totals(invoices, "actual_payment_date", months, book_start.to_period("M"), as_of)
    base_rev = float(rev.loc[[m for m in months if start.to_period("M") <= m < as_of.to_period("M")]].mean())

    # Fixed costs are sized off average turnover and held flat (headcount, loan, lease).
    staff_salary = round(0.05 * base_rev, -3)
    emi = round(0.03 * base_rev, -3)
    rent = round(0.02 * base_rev, -3)

    # Raw-material bills are paid in two tranches. The amount follows last month's
    # production (billing) but is partly throttled by last month's collections:
    # MSMEs pay their own vendors out of what customers pay them.
    rm_paid = {
        m: raw_material_share * (0.5 * rev[m - 1] + 0.5 * collected[m - 1]) * rng.uniform(0.95, 1.05)
        for m in months[1:]
    }

    rows = []

    def add(m: pd.Period, day: int, category: str, desc: str, amount: float, flexible: bool, tag: str) -> None:
        due = pd.Timestamp(year=m.year, month=m.month, day=day)
        if book_start <= due <= end and amount > 0:
            rows.append({
                "expense_id": f"EXP-{m.year}{m.month:02d}-{tag}",
                "category": category,
                "description": desc,
                "amount": round(float(amount), -2),
                "due_date": due,
                "is_flexible": flexible,
            })

    for m in months[2:]:
        prev_rev = rev[m - 1]
        add(m, 1, EXPENSE_PAYROLL, "Staff salaries", staff_salary * rng.uniform(0.98, 1.02), False, "PAY-STAFF")
        wages = 0.09 * base_rev * (0.85 + 0.15 * prev_rev / base_rev) * rng.uniform(0.97, 1.03)
        add(m, 7, EXPENSE_PAYROLL, "Shop-floor wages & overtime (Payment of Wages Act, by 7th)", wages, False, "PAY-WAGES")
        add(m, 5, EXPENSE_EMI_RENT, "Term-loan EMI (machinery)", emi, False, "EMI")
        add(m, 5, EXPENSE_EMI_RENT, "Factory shed rent", rent, False, "RENT")
        add(m, 10, EXPENSE_RAW_MATERIAL, "Steel / castings vendor payment (tranche 1)", 0.5 * rm_paid[m], True, "RM1")
        add(m, 25, EXPENSE_RAW_MATERIAL, "Steel / castings vendor payment (tranche 2)", 0.5 * rm_paid[m], True, "RM2")
        # GSTR-3B for month m-1: output GST on sales less input tax credit on purchases.
        output_gst = GST_FRACTION_OF_GROSS * prev_rev
        itc = GST_FRACTION_OF_GROSS * rm_paid[m - 1]
        add(m, 20, EXPENSE_GST_TAX, f"GSTR-3B net GST for {(m - 1).strftime('%b %Y')}", output_gst - itc, False, "GST")
        if m.month in (6, 9, 12, 3):
            add(m, 15, EXPENSE_GST_TAX, "Advance income tax instalment", 0.008 * 3 * base_rev, False, "ADVTAX")

    df = pd.DataFrame(rows).sort_values(["due_date", "expense_id"]).reset_index(drop=True)
    df["is_projected"] = df["due_date"] > as_of
    return df[EXPENSE_COLUMNS]


def _calibrate_raw_material_share(invoices: pd.DataFrame, start: pd.Timestamp, as_of: pd.Timestamp, seed: int) -> float:
    """Solve for the raw-material share that leaves a thin positive net margin over the history window.

    Thin margins keep the reconstructed cash history roughly stationary instead of
    trending, which is what a working-capital-stressed MSME looks like.
    """
    in_book = invoices["actual_payment_date"] >= _cashbook_start(start)
    collections = invoices.loc[in_book, "amount"].sum()
    target_outflow = collections * (1 - TARGET_NET_MARGIN)

    def hist_outflow(share: float) -> float:
        exp = _build_expenses(invoices, start, as_of, share, np.random.default_rng(seed + 1))
        return float(exp.loc[~exp["is_projected"], "amount"].sum())

    lo_share, hi_share = 0.40, 0.80
    lo, hi = hist_outflow(lo_share), hist_outflow(hi_share)
    # Outflow is (nearly) linear in the share; one secant step is enough.
    share = lo_share + (target_outflow - lo) * (hi_share - lo_share) / (hi - lo)
    return float(np.clip(share, 0.35, 0.80))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_msme_ledger(
    n_customers: int = 25,
    n_invoices: int = 500,
    start_date: str = "2025-01-01",
    seed: int = 42,
    as_of_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Generate a synthetic Indian MSME ledger.

    Args:
        n_customers: number of distinct customers (>= 3).
        n_invoices: number of sales invoices issued between start_date and as_of_date.
        start_date: first day of the ledger (YYYY-MM-DD).
        seed: RNG seed; identical arguments give identical output.
        as_of_date: evaluation date (YYYY-MM-DD). Defaults to start_date + 365 days.

    Returns:
        invoices_df: one row per invoice (InvoiceRecord fields plus helper columns).
            actual_payment_date is NaT for invoices unpaid at as_of_date (right-censored).
            delay_days is the survival duration from due date (to payment, or to as_of_date
            when censored) and event_observed is the event indicator.
        recurring_expenses_df: RecurringExpense rows from start_date to as_of_date + 90 days;
            is_projected marks obligations after as_of_date.
        starting_cash_balance: cash in hand (Rs) on as_of_date, i.e. the initial balance
            for the forward simulation.

    Both DataFrames carry the evaluation date in ``df.attrs["as_of_date"]``.
    """
    start = pd.Timestamp(start_date)
    as_of = pd.Timestamp(as_of_date) if as_of_date else start + pd.Timedelta(days=HISTORY_DAYS)
    if as_of <= start + pd.Timedelta(days=60):
        raise ValueError("as_of_date must be at least 60 days after start_date")

    rng = np.random.default_rng(seed)
    customers = _build_customers(n_customers, rng)
    invoices = _generate_invoices(customers, n_invoices, start, as_of, rng)

    rm_share = _calibrate_raw_material_share(invoices, start, as_of, seed)
    expenses = _build_expenses(invoices, start, as_of, rm_share, np.random.default_rng(seed + 1))

    for df in (invoices, expenses):
        df.attrs["as_of_date"] = as_of.strftime("%Y-%m-%d")
        df.attrs["cashbook_start_date"] = _cashbook_start(start).strftime("%Y-%m-%d")

    # Thin cash buffer, anchored to recent history: pick the as-of balance so that
    # roughly 5-15% of the last 90 days closed in deficit (a business skating along
    # its overdraft limit), clamped to 10-60% of a month's outflows. This keeps the
    # backtest's actual-deficit labels non-trivial and the next 60 days uncertain.
    hist = expenses.loc[~expenses["is_projected"]]
    monthly_outflow = hist.groupby(hist["due_date"].dt.to_period("M"))["amount"].sum().mean()
    relative = reconstruct_cash_history(invoices, expenses, 0.0)["closing_cash"].iloc[-90:]
    anchored = -float(relative.quantile(rng.uniform(0.05, 0.15)))
    starting_cash = float(round(np.clip(anchored, 0.10 * monthly_outflow, 0.60 * monthly_outflow), -3))
    return invoices, expenses, starting_cash


def reconstruct_cash_history(
    invoices_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    cash_as_of: float,
) -> pd.DataFrame:
    """Back out the daily closing cash balance from the cash-book start to the as-of date.

    cash(t) = cash(as_of) - sum of net flows on days (t, as_of]. Useful for
    backtesting (actual deficit indicators) and the ARIMA baseline.
    """
    as_of = pd.Timestamp(invoices_df.attrs["as_of_date"])
    start = pd.Timestamp(invoices_df.attrs.get("cashbook_start_date") or expenses_df["due_date"].min())
    days = pd.date_range(start, as_of, freq="D")
    paid = invoices_df[invoices_df["actual_payment_date"].between(start, as_of)]
    inflow = paid.groupby("actual_payment_date")["amount"].sum()
    hist = expenses_df[expenses_df["due_date"] <= as_of]
    outflow = hist.groupby("due_date")["amount"].sum()
    out = pd.DataFrame(index=days)
    out["inflow"] = inflow.reindex(days, fill_value=0.0)
    out["outflow"] = outflow.reindex(days, fill_value=0.0)
    out["net_flow"] = out["inflow"] - out["outflow"]
    future_net = out["net_flow"][::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    out["closing_cash"] = cash_as_of - future_net
    out.index.name = "date"
    return out


def _iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


def invoices_to_records(invoices_df: pd.DataFrame) -> List[InvoiceRecord]:
    """Convert invoice rows to schema dataclasses (dates as YYYY-MM-DD strings)."""
    return [
        InvoiceRecord(
            invoice_id=r.invoice_id,
            customer_id=r.customer_id,
            customer_segment=r.customer_segment,
            amount=float(r.amount),
            issue_date=_iso(r.issue_date),
            due_date=_iso(r.due_date),
            actual_payment_date=_iso(r.actual_payment_date),
            payment_terms_days=int(r.payment_terms_days),
            is_disputed=bool(r.is_disputed),
            customer_past_avg_delay=float(r.customer_past_avg_delay),
        )
        for r in invoices_df.itertuples(index=False)
    ]


def expenses_to_records(expenses_df: pd.DataFrame) -> List[RecurringExpense]:
    """Convert expense rows to schema dataclasses (dates as YYYY-MM-DD strings)."""
    return [
        RecurringExpense(
            expense_id=r.expense_id,
            category=r.category,
            amount=float(r.amount),
            due_date=_iso(r.due_date),
            is_flexible=bool(r.is_flexible),
        )
        for r in expenses_df.itertuples(index=False)
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _lakh(x: float) -> str:
    return f"Rs {x / 1e5:,.2f} L"


def _print_summary(invoices: pd.DataFrame, expenses: pd.DataFrame, cash: float) -> None:
    as_of = pd.Timestamp(invoices.attrs["as_of_date"])
    span_days = (as_of - invoices["issue_date"].min()).days
    turnover = invoices["amount"].sum() * 365 / span_days
    paid = invoices["event_observed"]
    open_inv = invoices[~paid]

    print("=" * 72)
    print("Runway synthetic MSME ledger")
    print("=" * 72)
    print(f"Ledger window          : {invoices['issue_date'].min():%Y-%m-%d} -> {as_of:%Y-%m-%d} (as-of)")
    print(f"Annualised turnover    : Rs {turnover / 1e7:,.2f} Cr")
    print(f"Invoices               : {len(invoices)} across {invoices['customer_id'].nunique()} customers")
    print(f"  paid (observed)      : {paid.sum()}")
    print(f"  open (censored)      : {(~paid).sum()}  worth {_lakh(open_inv['amount'].sum())}")
    print(f"  overdue at as-of     : {(open_inv['due_date'] < as_of).sum()}")
    print(f"  disputed             : {invoices['is_disputed'].sum()}")
    print(f"Starting cash (as-of)  : {_lakh(cash)}")

    print("\nPayment behaviour by segment (paid invoices, days past due):")
    seg = invoices[paid].groupby("customer_segment")["delay_days"].describe(percentiles=[0.5, 0.9])
    seg = seg[["count", "mean", "50%", "90%", "min", "max"]].round(1)
    counts = invoices.groupby("customer_segment").agg(customers=("customer_id", "nunique"),
                                                      avg_amount=("amount", "mean"))
    print(seg.join(counts).to_string())

    print("\nPayment terms mix:")
    print(invoices["payment_terms_days"].value_counts().sort_index().to_string())

    hist = expenses[~expenses["is_projected"]]
    n_months = hist["due_date"].dt.to_period("M").nunique()
    print("\nAverage monthly obligations (history):")
    print((hist.groupby("category")["amount"].sum() / n_months).map(_lakh).to_string())
    print(f"Projected obligations in next 60 days: "
          f"{_lakh(expenses[expenses['is_projected'] & (expenses['due_date'] <= as_of + pd.Timedelta(days=60))]['amount'].sum())}")

    cash_hist = reconstruct_cash_history(invoices, expenses, cash)
    last90 = cash_hist.iloc[-90:]
    print(f"\nReconstructed cash history: min {_lakh(cash_hist['closing_cash'].min())}, "
          f"max {_lakh(cash_hist['closing_cash'].max())}")
    print(f"Deficit days (cash < 0): {(cash_hist['closing_cash'] < 0).sum()} overall, "
          f"{(last90['closing_cash'] < 0).sum()} in the last 90 days")
    print("=" * 72)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    invoices_df, expenses_df, starting_cash = generate_msme_ledger()
    invoices_df.to_csv(out_dir / "sample_ledger.csv", index=False, date_format="%Y-%m-%d")
    expenses_df.to_csv(out_dir / "sample_expenses.csv", index=False, date_format="%Y-%m-%d")
    _print_summary(invoices_df, expenses_df, starting_cash)
    print(f"Saved {out_dir / 'sample_ledger.csv'}")
    print(f"Saved {out_dir / 'sample_expenses.csv'}")
