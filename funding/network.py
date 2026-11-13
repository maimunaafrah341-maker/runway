"""
The lending network: businesses with spare cash, the conditions they lend on, and the requests between
them and a business that needs cash.

Each demo link has its own LendingNetwork (DemoRooms, keyed by the link's ?room= code and kept in
st.cache_resource), so a request Raghav sends appears in the lender's window within seconds and the answer
comes back, while other people trying the same public URL get their own private demo and never see or
change it. It lives in memory, so a restart clears it; that suits a demo, and a real deployment would keep
it in a database.

Runway never lends, holds or moves money here. It shares the borrower's cash-flow check (with consent),
drafts the terms, and the two companies sign and pay each other directly.
"""
from __future__ import annotations

import itertools
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

WAITING, ACCEPTED, DECLINED, COUNTERED, WITHDRAWN = "waiting", "accepted", "declined", "countered", "withdrawn"
OPEN_STATUSES = (WAITING, COUNTERED)


@dataclass(frozen=True)
class Lender:
    lid: str
    company: str
    contact: str
    city: str
    business: str
    why_surplus: str
    available: float          # rupees it is willing to lend in total
    max_days: int             # longest loan it makes
    min_rate: float           # lowest interest it accepts, a year
    needs_security: bool      # wants unpaid invoices pledged
    min_on_time: float        # lowest Runway on-time score it accepts (0 = does not ask)


# Fictional demo companies in the same auto-parts cluster; none of them is a real business.
DEMO_LENDERS: Tuple[Lender, ...] = (
    Lender("ganga", "Ganga Castings Pvt Ltd", "Meera Rao", "Coimbatore", "Aluminium castings",
           "Two OEMs paid a quarter's invoices early and the next furnace upgrade is in March.",
           3_200_000, 45, 0.110, True, 0.70),
    Lender("pennar", "Pennar Polymers Pvt Ltd", "Arjun Reddy", "Nellore", "Plastic auto parts",
           "A customer paid an advance and a machine purchase moved to February.",
           1_500_000, 30, 0.115, False, 0.65),
    Lender("yamuna", "Yamuna Rubber Pvt Ltd", "Farhan Qureshi", "Faridabad", "Rubber seals and hoses",
           "Festive-season orders are paid and the firm keeps a cash reserve.",
           4_000_000, 60, 0.135, True, 0.75),
    Lender("palar", "Palar Packaging Pvt Ltd", "Kavya Menon", "Vellore", "Packaging for auto parts",
           "Quarterly surplus sits in a low-interest sweep account.",
           1_000_000, 21, 0.105, False, 0.60),
)


@dataclass(frozen=True)
class LoanRequest:
    rid: str
    lender_id: str
    borrower: str                 # company name
    owner: str
    borrower_note: str            # one line: what the business is
    amount: float
    rate: float                   # offered, a year
    start: pd.Timestamp
    repay: pd.Timestamp
    interest: float
    security: tuple = ()          # pledged invoices: ({invoice_id, customer, amount, due_date, p_paid}, ...)
    security_total: float = 0.0
    security_cover: Optional[float] = None
    check: Optional[dict] = None  # Runway's cash-flow check, None if the borrower chose not to share it
    status: str = WAITING
    counter_rate: Optional[float] = None
    agreed_rate: Optional[float] = None
    note: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def days(self) -> int:
        return int((self.repay - self.start).days)

    @property
    def rate_now(self) -> float:
        """The rate on the table: agreed, else countered, else offered."""
        return self.agreed_rate or self.counter_rate or self.rate

    @property
    def interest_now(self) -> float:
        return self.amount * self.rate_now * self.days / 365


def fit_checks(lender: Lender, days: int, rate: float, security_offered: bool,
               on_time: Optional[float]) -> List[Tuple[bool, str]]:
    """Each of the lender's conditions, and whether this request meets it."""
    checks = [
        (days <= lender.max_days, f"Lends for up to {lender.max_days} days"),
        (rate >= lender.min_rate - 1e-9, f"Wants at least {lender.min_rate:.1%} a year"),
    ]
    if lender.needs_security:
        checks.append((security_offered, "Wants unpaid invoices pledged as security"))
    if lender.min_on_time > 0:
        checks.append((on_time is not None and on_time >= lender.min_on_time - 1e-9,
                       f"Wants a Runway on-time score of at least {lender.min_on_time:.0%}"))
    return checks


def fits(lender: Lender, days: int, rate: float, security_offered: bool, on_time: Optional[float]) -> bool:
    return lender.available > 0 and all(ok for ok, _ in fit_checks(lender, days, rate, security_offered, on_time))


def split_amount(amount: float, lenders: Sequence[Lender], step: float = 10_000) -> List[Tuple[Lender, float]]:
    """Fill the request from the cheapest lenders first; each gives up to what it has available.

    Shares are whole multiples of `step` except the last one, which takes the exact remainder.
    """
    parts, left = [], float(amount)
    for lender in sorted(lenders, key=lambda x: (x.min_rate, -x.available)):
        if left <= 0:
            break
        part = min(left, float(lender.available))
        if part < left:
            part = (part // step) * step
        if part > 0:
            parts.append((lender, part))
            left -= part
    return parts


class LendingNetwork:
    """Thread-safe, in-memory store shared by every session. All reads return immutable snapshots."""

    def __init__(self, lenders: Sequence[Lender] = DEMO_LENDERS) -> None:
        self._lock = threading.RLock()
        self._seed = tuple(lenders)
        self._ids = itertools.count(1)    # never restarts, so a reset never reuses a request id
        self.version = 0
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._lenders: Dict[str, Lender] = {x.lid: x for x in self._seed}
            self._requests: Dict[str, LoanRequest] = {}
            self.version += 1

    # Lenders --------------------------------------------------------------- #
    def lenders(self) -> List[Lender]:
        with self._lock:
            return list(self._lenders.values())

    def lender(self, lid: str) -> Lender:
        with self._lock:
            return self._lenders[lid]

    def set_conditions(self, lid: str, **changes) -> Lender:
        allowed = {"available", "max_days", "min_rate", "needs_security", "min_on_time"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"cannot change {sorted(bad)}")
        with self._lock:
            self._lenders[lid] = replace(self._lenders[lid], **changes)
            self.version += 1
            return self._lenders[lid]

    # Requests -------------------------------------------------------------- #
    def send(self, **fields) -> LoanRequest:
        with self._lock:
            if fields["lender_id"] not in self._lenders:
                raise KeyError(fields["lender_id"])
            req = LoanRequest(rid=f"RQ-{next(self._ids):04d}", **fields)
            self._requests[req.rid] = req
            self.version += 1
            return req

    def request(self, rid: str) -> LoanRequest:
        with self._lock:
            return self._requests[rid]

    def requests(self, lender_id: Optional[str] = None, borrower: Optional[str] = None) -> List[LoanRequest]:
        """Newest first."""
        with self._lock:
            out = [r for r in self._requests.values()
                   if (lender_id is None or r.lender_id == lender_id) and (borrower is None or r.borrower == borrower)]
        return sorted(out, key=lambda r: r.created, reverse=True)

    def _update(self, rid: str, allowed_from: Sequence[str], **changes) -> LoanRequest:
        with self._lock:
            req = self._requests[rid]
            if req.status not in allowed_from:
                raise ValueError(f"{rid} is {req.status}; cannot change it now")
            req = replace(req, updated=time.time(), **changes)
            self._requests[rid] = req
            self.version += 1
            return req

    def respond(self, rid: str, decision: str, counter_rate: Optional[float] = None, note: str = "") -> LoanRequest:
        """The lender's answer: 'accept', 'decline', or 'counter' with a higher rate."""
        if decision == "accept":
            with self._lock:
                req = self._requests[rid]
                lender = self._lenders[req.lender_id]
                if req.amount > lender.available + 1e-6:
                    raise ValueError(f"{lender.company} has only {lender.available:,.0f} available")
                out = self._update(rid, (WAITING,), status=ACCEPTED, agreed_rate=req.rate, note=note)
                self._lenders[lender.lid] = replace(lender, available=lender.available - req.amount)
                return out
        if decision == "decline":
            return self._update(rid, (WAITING,), status=DECLINED, note=note)
        if decision == "counter":
            req = self.request(rid)
            if counter_rate is None or counter_rate <= req.rate:
                raise ValueError("a counter-offer must ask for a higher rate than offered")
            return self._update(rid, (WAITING,), status=COUNTERED, counter_rate=float(counter_rate), note=note)
        raise ValueError(f"unknown decision {decision!r}")

    def answer_counter(self, rid: str, accept: bool) -> LoanRequest:
        """The borrower's answer to a counter-offer."""
        if not accept:
            return self._update(rid, (COUNTERED,), status=DECLINED, note="Declined the counter-offer")
        with self._lock:
            req = self._requests[rid]
            lender = self._lenders[req.lender_id]
            if req.amount > lender.available + 1e-6:
                raise ValueError(f"{lender.company} has only {lender.available:,.0f} available")
            out = self._update(rid, (COUNTERED,), status=ACCEPTED, agreed_rate=req.counter_rate)
            self._lenders[lender.lid] = replace(lender, available=lender.available - req.amount)
            return out

    def withdraw(self, rid: str) -> LoanRequest:
        return self._update(rid, OPEN_STATUSES, status=WITHDRAWN)


# --------------------------------------------------------------------------- #
# Demo rooms: one private network per demo link, so visitors to a public URL never share requests
# --------------------------------------------------------------------------- #
_ROOM_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"      # no 0/o, 1/l/i: easy to read out and type
_ROOM_PATTERN = re.compile(r"[a-z0-9]{3,16}")


def new_room_code(length: int = 6) -> str:
    return "".join(secrets.choice(_ROOM_ALPHABET) for _ in range(length))


def valid_room(code: Optional[str]) -> bool:
    return bool(code) and _ROOM_PATTERN.fullmatch(code) is not None


class DemoRooms:
    """A LendingNetwork per room code. Windows opened with the same code see the same requests; any other
    code (another visitor) gets a fresh, empty network. Beyond `max_rooms`, the least recently used go."""

    def __init__(self, max_rooms: int = 500):
        self.max_rooms = max_rooms
        self._lock = threading.Lock()
        self._rooms: "OrderedDict[str, LendingNetwork]" = OrderedDict()

    def get(self, code: str) -> LendingNetwork:
        if not valid_room(code):
            raise ValueError(f"not a room code: {code!r}")
        with self._lock:
            net = self._rooms.pop(code, None)
            if net is None:
                net = LendingNetwork()
            self._rooms[code] = net
            while len(self._rooms) > self.max_rooms:
                self._rooms.popitem(last=False)
            return net

    def __len__(self) -> int:
        with self._lock:
            return len(self._rooms)

    def __contains__(self, code: str) -> bool:
        with self._lock:
            return code in self._rooms


# --------------------------------------------------------------------------- #
# Calendar file: the loan's dates as all-day events with a reminder, for any phone or desktop calendar
# --------------------------------------------------------------------------- #
def _ics_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def calendar_ics(events: Sequence[Tuple[pd.Timestamp, str, str]], name: str = "Runway loan plan") -> str:
    """An iCalendar (.ics) file with one all-day event per (date, title, description), each with a
    reminder the day before at 9:00."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Runway//Loan plan//EN", "CALSCALE:GREGORIAN",
             f"X-WR-CALNAME:{_ics_text(name)}"]
    for i, (when, title, desc) in enumerate(events):
        d = pd.Timestamp(when)
        lines += [
            "BEGIN:VEVENT", f"UID:runway-{d:%Y%m%d}-{i}@runway.local", f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{d:%Y%m%d}", f"DTEND;VALUE=DATE:{d + pd.Timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{_ics_text(title)}", f"DESCRIPTION:{_ics_text(desc)}",
            "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_ics_text(title)}", "TRIGGER:-PT15H", "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# --------------------------------------------------------------------------- #
# Term sheet (plain text, both sides can download it once a request is accepted)
# --------------------------------------------------------------------------- #
def _rupees(value: float) -> str:
    a = abs(value)
    if a >= 1e7:
        return f"Rs {a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"Rs {a / 1e5:.2f} lakh"
    return f"Rs {a:,.0f}"


def term_sheet(req: LoanRequest, lender: Lender) -> str:
    rate = req.rate_now
    lines = [
        "SHORT-TERM LOAN: SUMMARY OF AGREED TERMS (draft for both companies to sign)",
        f"Reference {req.rid} · prepared by Runway · demo data, not legal advice",
        "",
        f"Lender     {lender.company}, {lender.city} (contact: {lender.contact})",
        f"Borrower   {req.borrower} (contact: {req.owner})",
        "",
        f"Amount     {_rupees(req.amount)}",
        f"Interest   {rate:.2%} a year, simple, actual/365",
        f"Paid out   {req.start:%d %b %Y}, by RTGS/NEFT from the lender's bank account to the borrower's",
        f"Repay by   {req.repay:%d %b %Y} ({req.days} days): {_rupees(req.amount)} + interest "
        f"{_rupees(req.interest_now)} = {_rupees(req.amount + req.interest_now)}",
        "Early repayment allowed; interest only for the days used.",
    ]
    if req.security:
        lines += ["", f"Security   these unpaid invoices ({_rupees(req.security_total)}) are pledged to the lender "
                      "until repayment:"]
        lines += [f"           {s['invoice_id']}  {s['customer']}  {_rupees(s['amount'])}  due {s['due_date']:%d %b %Y}"
                  for s in req.security]
    if req.check:
        lines += ["", f"Runway on-time score shared by the borrower: the full repayment is in the bank on the date "
                      f"in {req.check['p_repay_on_time']:.0%} of {req.check['n_futures']:,} simulated futures."]
    lines += [
        "",
        "Steps",
        "1. Lender: board resolution approving the loan, passed with the consent of all directors present "
        "(Companies Act 2013, Sec 186). The rate must not be below the government-bond yield for a similar "
        "period (Sec 186(7)).",
        "2. Both: sign a loan agreement. If invoices are pledged, the borrower registers the charge with the "
        "Registrar of Companies within 30 days (Sec 77).",
        "3. A loan from one company to another is not a public deposit (Companies (Acceptance of Deposits) "
        "Rules 2014, Rule 2(1)(c)).",
        "4. Lender transfers the money; borrower repays on or before the date above.",
        "",
        "Runway does not lend, hold or move any money. It shared the borrower's cash-flow check with consent.",
    ]
    return "\n".join(lines)
