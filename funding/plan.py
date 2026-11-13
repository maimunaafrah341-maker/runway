"""
How much credit Raghav needs, for how long, how safe it is to lend to him, and where he can get it.

Everything here works on the simulator's own arrays, so a loan is checked on the same 5,000 futures as
the forecast:
    trajectories   (n_sims, horizon)  bank balance at the end of each simulated day, no new credit
    payment_days   (n_sims, n_open)   day index on which each unpaid invoice is paid (horizon = not paid)
Day index 0 is the day after the ledger date.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

YEAR_DAYS = 365
AFTER_REPAY_DAYS = 7          # a repayment that empties the account the week after is not a safe date
SECURITY_MARGIN = 1.25        # pledge invoices worth 125% of what is owed, the usual margin on receivables
TREDS_RATE = 0.10             # typical TReDS discount for a large, well-rated buyer (estimate)
TREDS_SEGMENTS = ("Tier-1 Auto OEM",)


def interest(amount: float, rate: float, days: int) -> float:
    """Simple interest, actual/365, the usual basis for a short business loan in India.

    A negative amount (money lent out) gives negative interest, i.e. interest received.
    """
    return float(amount) * float(rate) * max(0, int(days)) / YEAR_DAYS


def loan_path(trajectories: np.ndarray, amount: float, start: int, repay: int, rate: float) -> np.ndarray:
    """Simulated balances with the loan: +amount on day `start`, −(amount + interest) on day `repay`."""
    t = np.arange(trajectories.shape[1])[None, :]
    cash = trajectories + amount * (t >= start)
    return cash - (amount + interest(amount, rate, repay - start)) * (t >= repay)


@dataclass(frozen=True)
class LoanCheck:
    amount: float
    start: int                 # day index the money arrives
    repay: int                 # day index it is paid back
    rate: float                # a year
    interest: float
    p_repay_on_time: float     # has the full repayment in the bank on the repay day
    p_short_without: float     # balance goes below ₹0 between today and a week after the repay day, no loan
    p_short_with: float        # the same with the loan in and the repayment out
    horizon: int               # days covered by the two shortfall numbers

    @property
    def days(self) -> int:
        return self.repay - self.start


def check_loan(trajectories: np.ndarray, amount: float, start: int, repay: int, rate: float,
               after_days: int = AFTER_REPAY_DAYS) -> LoanCheck:
    n_days = trajectories.shape[1]
    if not 0 <= start < repay < n_days:
        raise ValueError(f"need 0 <= start < repay < {n_days}, got start={start}, repay={repay}")
    horizon = min(n_days, repay + after_days)
    owed_interest = interest(amount, rate, repay - start)
    with_loan = loan_path(trajectories, amount, start, repay, rate)[:, :horizon]
    return LoanCheck(
        amount=float(amount), start=start, repay=repay, rate=float(rate), interest=owed_interest,
        p_repay_on_time=float((trajectories[:, repay] >= owed_interest).mean()),
        p_short_without=float((trajectories[:, :horizon] < 0).any(axis=1).mean()),
        p_short_with=float((with_loan < 0).any(axis=1).mean()),
        horizon=horizon,
    )


def suggest_repay_day(trajectories: np.ndarray, amount: float, start: int, rate: float,
                      min_days: int = 14, max_days: int = 45, after_days: int = AFTER_REPAY_DAYS) -> int:
    """The repay day that keeps the most futures above ₹0 from the loan's start to a week after repaying.

    Repaying just before the next wage or raw-material day would only move the crunch, so the week after
    the repayment counts too. Ties go to the earliest day (the cheapest loan).
    """
    n_days = trajectories.shape[1]
    last = min(start + max_days, n_days - 1 - after_days)
    first = min(start + min_days, last)
    if first <= start:
        raise ValueError("not enough simulated days after the start for a loan")
    best, best_score = first, -1.0
    for repay in range(first, last + 1):
        path = loan_path(trajectories, amount, start, repay, rate)[:, start:repay + after_days + 1]
        score = float((path >= 0).all(axis=1).mean())
        if score > best_score + 1e-9:
            best, best_score = repay, score
    return best


def apply_loans(trajectories: np.ndarray, loans: Iterable[tuple]) -> np.ndarray:
    """Simulated balances with several agreed loans, each (amount, start, repay, rate).

    A positive amount is borrowed (in on `start`, out with interest on `repay`); a negative amount is lent
    (out on `start`, back with interest on `repay`).
    """
    t = np.arange(trajectories.shape[1])[None, :]
    cash = trajectories.astype(float, copy=True)
    for amount, start, repay, rate in loans:
        cash += amount * (t >= start) - (amount + interest(amount, rate, repay - start)) * (t >= repay)
    return cash


def safe_to_lend(trajectories: np.ndarray, days: int, keep: float, quantile: float = 10) -> float:
    """How much a business can lend for `days` days and still keep `keep` rupees in the bank in 9 of 10 futures.

    Lending A lowers every day's balance by A, so the 10th percentile of the lowest balance over the period
    minus `keep` is the largest A that stays safe (the loan coming back early only helps).
    """
    days = max(1, min(int(days), trajectories.shape[1]))
    low = float(np.percentile(trajectories[:, :days].min(axis=1), quantile))
    return max(0.0, low - float(keep))


def lending_effect(trajectories: np.ndarray, amount: float, start: int, repay: int, rate: float,
                   horizon: Optional[int] = None) -> tuple:
    """The lender's own shortfall risk without and with making this loan (money out on `start`, back with
    interest on `repay`), on the lender's own simulated futures."""
    horizon = trajectories.shape[1] if horizon is None else min(horizon, trajectories.shape[1])
    t = np.arange(trajectories.shape[1])[None, :]
    lent = (trajectories - amount * (t >= start)
            + (amount + interest(amount, rate, repay - start)) * (t >= repay))[:, :horizon]
    return (float((trajectories[:, :horizon] < 0).any(axis=1).mean()), float((lent < 0).any(axis=1).mean()))


def credit_dips(median_path: Sequence[float], dates: Sequence) -> List[tuple]:
    """Stretches of days where the typical balance is below ₹0: [(first_date, last_date), ...]."""
    neg = np.asarray(median_path) < 0
    dips, i = [], 0
    while i < len(neg):
        if neg[i]:
            j = i
            while j + 1 < len(neg) and neg[j + 1]:
                j += 1
            dips.append((pd.Timestamp(dates[i]), pd.Timestamp(dates[j])))
            i = j + 1
        else:
            i += 1
    return dips


# --------------------------------------------------------------------------- #
# Security: unpaid invoices pledged to a lender
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Security:
    invoices: tuple            # ({invoice_id, customer, amount, due_date, p_paid}, ...)
    total: float               # face value pledged
    cover: float               # share of futures in which the pledged invoices alone pay what is owed


def pick_security(open_invoices: pd.DataFrame, payment_days: np.ndarray, repay: int, owed: float,
                  exclude: Iterable[str] = (), margin: float = SECURITY_MARGIN) -> Security:
    """Pledge the invoices most likely to be paid by the repay day until they are worth `margin` × owed.

    `open_invoices` rows must be in the same order as the columns of `payment_days`.
    """
    if len(open_invoices) != payment_days.shape[1]:
        raise ValueError("open_invoices and payment_days do not line up")
    df = open_invoices.reset_index(drop=True).copy()
    df["col"] = np.arange(len(df))
    df["p_paid"] = (payment_days <= repay).mean(axis=0)
    skip = set(exclude)
    disputed = (df["is_disputed"].astype(bool) if "is_disputed" in df
                else pd.Series(False, index=df.index))
    cand = df[~df["invoice_id"].isin(skip) & ~disputed & (df["p_paid"] > 0)]
    cand = cand.sort_values(["p_paid", "amount"], ascending=[False, False])
    picked, total = [], 0.0
    for row in cand.itertuples(index=False):
        if total >= margin * owed:
            break
        picked.append(row)
        total += float(row.amount)
    if not picked:
        return Security(invoices=(), total=0.0, cover=0.0)
    cols = [r.col for r in picked]
    amounts = np.array([float(r.amount) for r in picked])
    collected = ((payment_days[:, cols] <= repay) * amounts[None, :]).sum(axis=1)
    rows = tuple({"invoice_id": r.invoice_id, "customer": r.customer_name, "amount": float(r.amount),
                  "due_date": pd.Timestamp(r.due_date), "p_paid": float(r.p_paid)} for r in picked)
    return Security(invoices=rows, total=total, cover=float((collected >= owed).mean()))


# --------------------------------------------------------------------------- #
# Where the money can come from
# --------------------------------------------------------------------------- #
def treds_invoices(open_invoices: pd.DataFrame, as_of: pd.Timestamp,
                   segments: Sequence[str] = TREDS_SEGMENTS) -> pd.DataFrame:
    """Unpaid, undisputed, not-yet-due invoices on large buyers: the ones a TReDS financier can buy."""
    df = open_invoices.copy()
    df["due_date"] = pd.to_datetime(df["due_date"])
    keep = (df["customer_segment"].isin(segments) & ~df["is_disputed"].astype(bool) & (df["due_date"] > as_of))
    return df[keep].sort_values("due_date")


def treds_discount(invoices: pd.DataFrame, sell_date: pd.Timestamp, rate: float = TREDS_RATE) -> float:
    """What the financier keeps: interest from the day the invoice is sold to its due date."""
    days = (pd.to_datetime(invoices["due_date"]) - sell_date).dt.days.clip(lower=0)
    return float((invoices["amount"] * rate * days / YEAR_DAYS).sum())


@dataclass(frozen=True)
class CreditSource:
    key: str
    name: str
    how: str                    # one line: what it is
    capacity: Optional[float]   # rupees available, None = decided by the lender
    rate_low: float
    rate_high: float
    days_low: int               # working days until the money is in the bank
    days_high: int
    needs: str
    cost: float                 # for this loan, at the middle of the rate range
    fits: str                   # "yes", "maybe", "no"
    fits_note: str


def credit_sources(need: float, days_left: int, tenor: int, as_of: pd.Timestamp,
                   treds: pd.DataFrame, peer_total: float, peer_rates: Sequence[float],
                   peer_offer_rate: Optional[float] = None) -> List[CreditSource]:
    """The realistic ways an Indian MSME can raise `need` rupees within `days_left` days.

    Rates and timings are typical market ranges (estimates for a sound small manufacturer), not quotes.
    `peer_rates` are the matching lenders' minimum rates; the peer cost uses `peer_offer_rate` if given.
    """
    def mid(lo: float, hi: float) -> float:
        return (lo + hi) / 2

    def fit(days_high: int, days_low: int) -> tuple:
        if days_high <= days_left:
            return "yes", f"Money in {days_low}–{days_high} days"
        if days_low <= days_left:
            return "maybe", f"Only if you apply today ({days_low}–{days_high} days)"
        return "no", f"Takes {days_low}–{days_high} days"

    rows: List[CreditSource] = []

    treds_total = float(treds["amount"].sum()) if len(treds) else 0.0
    sold = treds
    if treds_total > need:   # sell only as many invoices as needed, earliest due first
        sold = treds[treds["amount"].cumsum().shift(fill_value=0.0) < need]
    rows.append(CreditSource(
        key="treds", name="Sell invoices on TReDS",
        how="A bank or NBFC on an RBI-licensed TReDS exchange (RXIL, M1xchange, Invoicemart) pays you now "
            "for unpaid invoices on big buyers; the buyer pays them on the due date. Not a loan.",
        capacity=treds_total, rate_low=0.08, rate_high=0.11, days_low=1, days_high=2,
        needs="A TReDS account (Udyam + KYC, about a week to open) and the buyer accepting each invoice on "
              "the exchange. Companies with turnover above ₹250 Cr must be on TReDS.",
        cost=treds_discount(sold, as_of + pd.Timedelta(days=1)),
        fits="maybe", fits_note="If you already have an account",
    ))

    if peer_total > 0:
        lo, hi = (min(peer_rates), max(peer_rates)) if peer_rates else (0.11, 0.13)
        f, note = fit(2, 1)
        rows.append(CreditSource(
            key="peer", name="Borrow from a business with spare cash",
            how="A company in the Runway network with idle cash lends to you for a few weeks, after seeing "
                "Runway's check on your cash flow. A direct company-to-company loan.",
            capacity=peer_total, rate_low=lo, rate_high=hi, days_low=1, days_high=2,
            needs="A short loan agreement; the lending company's board approves it (Companies Act Sec 186). "
                  "Usually unpaid invoices pledged as security.",
            cost=interest(min(need, peer_total), peer_offer_rate or mid(lo, hi), tenor), fits=f, fits_note=note,
        ))

    f, note = fit(7, 3)
    rows.append(CreditSource(
        key="bank", name="Bank overdraft or cash-credit top-up",
        how="Ask your bank to raise your cash-credit limit or give a short working-capital loan. "
            "CGTMSE guarantees cover micro and small firms up to ₹10 Cr, so no property is needed.",
        capacity=None, rate_low=0.095, rate_high=0.12, days_low=3, days_high=7,
        needs="6–12 months of bank statements, GST returns, last audited accounts, Udyam certificate.",
        cost=interest(need, mid(0.095, 0.12), tenor), fits=f, fits_note=note,
    ))

    f, note = fit(3, 2)
    rows.append(CreditSource(
        key="nbfc", name="NBFC or fintech working-capital loan",
        how="An online lender sanctions a short unsecured loan from your GST and bank data.",
        capacity=None, rate_low=0.15, rate_high=0.24, days_low=2, days_high=3,
        needs="GST login, bank statements, PAN and KYC. Fast, but costs about twice the bank.",
        cost=interest(need, mid(0.15, 0.24), tenor), fits=f, fits_note=note,
    ))
    return rows
