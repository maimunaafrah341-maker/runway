"""
MSMED Act interest on an unpaid invoice (Sections 2(b), 15 and 16).

* The legal due date is the agreed credit period, capped at 45 days from acceptance (Section 15 proviso).
  Invoices carry written payment terms, so the 15-day default in Section 2(b) is not used.
* Acceptance is taken as the invoice date (deemed acceptance on delivery, no written objection).
  For a disputed invoice the clock may start later: Section 2(b)(i)(b). The result flags that case.
* Interest is compounded monthly at three times the RBI bank rate, from the day after the legal due date.
  Part months are compounded pro rata, so the figure is an estimate, not a legal computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from assistant.legal_corpus import MAX_CREDIT_DAYS, MSMED_RATE

DAYS_PER_MONTH = 365 / 12


@dataclass(frozen=True)
class InterestLine:
    invoice_id: str
    customer: str
    amount: float
    issue_date: pd.Timestamp
    agreed_terms_days: int
    legal_due_date: pd.Timestamp       # issue date + min(agreed terms, 45)
    days_late: int                     # days past the legal due date as of the as-of date
    interest: float                    # rupees accrued so far
    terms_over_limit: bool             # agreed terms longer than the 45 days the law allows
    disputed: bool


def msmed_interest(amount: float, days_late: int, annual_rate: float = MSMED_RATE) -> float:
    """Compound interest with monthly rests for `days_late` days."""
    if days_late <= 0 or amount <= 0:
        return 0.0
    return float(amount * ((1 + annual_rate / 12) ** (days_late / DAYS_PER_MONTH) - 1))


def interest_for_invoices(invoices: pd.DataFrame, as_of: pd.Timestamp,
                          annual_rate: float = MSMED_RATE) -> List[InterestLine]:
    """One line per unpaid invoice in `invoices`, most interest first."""
    as_of = pd.Timestamp(as_of)
    open_inv = invoices[invoices["actual_payment_date"].isna()]
    lines = []
    for r in open_inv.itertuples(index=False):
        issue = pd.Timestamp(r.issue_date)
        terms = int(r.payment_terms_days)
        legal_due = issue + pd.Timedelta(days=min(terms, MAX_CREDIT_DAYS))
        late = max(0, (as_of - legal_due).days)
        lines.append(InterestLine(
            invoice_id=r.invoice_id, customer=r.customer_name, amount=float(r.amount), issue_date=issue,
            agreed_terms_days=terms, legal_due_date=legal_due, days_late=late,
            interest=msmed_interest(float(r.amount), late, annual_rate),
            terms_over_limit=terms > MAX_CREDIT_DAYS, disputed=bool(r.is_disputed),
        ))
    lines.sort(key=lambda x: (-x.interest, -x.amount))
    return lines
