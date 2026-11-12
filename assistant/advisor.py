"""
Runway legal assistant: retrieval-augmented answers about late payments (MSMED Act 2006).

For every question:
1. Retrieve  - BM25 search over the Act's text and a few dated notes (assistant.retriever). Offline.
2. Ground    - pull facts from the business's own ledger: unpaid invoices of any customer named in the
               question, the Section 16 interest on them (assistant.interest) and the cash forecast.
3. Generate  - online with an API key: Claude (Anthropic) or Gemini (Google) writes the answer from those
               passages and facts only, citing sections, in the language of the question.
               Offline or without a key: a template answer built from the same passages and facts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Sequence

import pandas as pd
import requests

from assistant.interest import InterestLine, interest_for_invoices
from assistant.legal_corpus import (BANK_RATE, BANK_RATE_CHECKED, MAX_CREDIT_DAYS, MSMED_RATE, PASSAGE_BY_ID,
                                    Passage)
from assistant.retriever import LegalRetriever, concepts

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_LABEL = "Claude Opus 5.5"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"    # marked "Stable" on Google's model list, Nov 2026
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
SMALL_TURNOVER_LIMIT = 100e7            # ₹100 crore (limits from 1 April 2025)
MICRO_TURNOVER_LIMIT = 10e7             # ₹10 crore

_retriever = LegalRetriever()


def inr(value: float) -> str:
    a = abs(value)
    sign = "−" if value < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    return f"{sign}₹{a:,.0f}"


def _d(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y")


# --------------------------------------------------------------------------- #
# Step 2: facts from the business's own data
# --------------------------------------------------------------------------- #
@dataclass
class BusinessContext:
    company: str
    owner: str
    as_of: pd.Timestamp
    turnover: float
    invoices: pd.DataFrame
    p_short_30d: float
    crunch_window: str
    risk_drivers: List[dict] = field(default_factory=list)   # sim.top_risk_invoices


@dataclass
class Facts:
    text: str                                   # sent to Claude and shown under "Numbers used"
    customers: List[str]
    lines_by_customer: Dict[str, List[InterestLine]]
    all_lines: List[InterestLine]
    size_class: str
    turnover: float


def find_customers(question: str, names: Sequence[str]) -> List[str]:
    """Customers named in the question, matched on the distinctive first word of the name."""
    q = question.lower()
    return [n for n in names if re.search(rf"\b{re.escape(n.split()[0].lower())}\b", q)]


def _size_class(turnover: float) -> str:
    if turnover <= MICRO_TURNOVER_LIMIT:
        return "micro"
    if turnover <= SMALL_TURNOVER_LIMIT:
        return "small"
    return "medium or larger"


def _line_text(x: InterestLine) -> str:
    bits = [f"{x.invoice_id}: {inr(x.amount)}, invoiced {_d(x.issue_date)}, agreed terms {x.agreed_terms_days} days"]
    if x.terms_over_limit:
        bits.append(f" (longer than the {MAX_CREDIT_DAYS}-day legal limit)")
    bits.append(f", legal due date {_d(x.legal_due_date)}")
    if x.days_late:
        bits.append(f", {x.days_late} days late, Section 16 interest so far {inr(x.interest)}")
    else:
        bits.append(", not yet past the legal due date")
    if x.disputed:
        bits.append(", marked DISPUTED in the ledger")
    return "".join(bits)


def build_facts(question: str, ctx: BusinessContext) -> Facts:
    lines = interest_for_invoices(ctx.invoices, ctx.as_of)
    names = list(dict.fromkeys(ctx.invoices["customer_name"]))
    named = find_customers(question, names)
    by_cust: Dict[str, List[InterestLine]] = {}
    for x in lines:
        by_cust.setdefault(x.customer, []).append(x)

    late = [x for x in lines if x.days_late > 0]
    long_terms = [x for x in lines if x.terms_over_limit]
    size = _size_class(ctx.turnover)
    out = [
        f"Business: {ctx.company} (owner {ctx.owner}), an auto-components manufacturer. Ledger date: {_d(ctx.as_of)}.",
        f"Turnover in the last 12 months: {inr(ctx.turnover)}, which is within the {size}-enterprise turnover "
        f"limit. Investment in plant and machinery and Udyam registration status are not in the data.",
        f"Section 16 rate used: {MSMED_RATE:.2%} a year compounded monthly (3 x RBI Bank Rate {BANK_RATE:.2%}, "
        f"checked {BANK_RATE_CHECKED}). Interest figures are estimates, assuming acceptance on the invoice date.",
        f"Cash forecast: {ctx.p_short_30d:.0%} chance cash runs short in the next 30 days, most likely {ctx.crunch_window}.",
        f"Unpaid invoices: {len(lines)} worth {inr(sum(x.amount for x in lines))}. Past the legal due date: "
        f"{len(late)} worth {inr(sum(x.amount for x in late))}, with Section 16 interest so far "
        f"{inr(sum(x.interest for x in late))}.",
        f"Unpaid invoices with credit terms longer than {MAX_CREDIT_DAYS} days: {len(long_terms)}.",
    ]
    for name in named:
        cl = by_cust.get(name, [])
        if not cl:
            out.append(f"Customer {name}: no unpaid invoices.")
            continue
        out.append(f"Customer {name}, unpaid invoices:")
        out += [f"- {_line_text(x)}" for x in cl]
        out.append(f"Total for {name}: invoices {inr(sum(x.amount for x in cl))}, "
                   f"interest so far {inr(sum(x.interest for x in cl))}.")
        for d in ctx.risk_drivers:
            if d.get("customer") == name:
                out.append(f"Forecast link: if {d['invoice_id']} is paid on time, the 30-day shortfall risk falls "
                           f"by {d['marginal_risk_reduction'] * 100:.1f} points.")
    if not named:
        out.append("Largest interest claims in the ledger:")
        out += [f"- {x.customer}, {_line_text(x)}" for x in late[:6]]
    return Facts(text="\n".join(out), customers=named, lines_by_customer=by_cust, all_lines=lines,
                 size_class=size, turnover=ctx.turnover)


# --------------------------------------------------------------------------- #
# Step 1: retrieval
# --------------------------------------------------------------------------- #
# Passages every answer on a topic relies on (the offline templates cite them), added to the search hits
# so Claude always sees them too.
_ELIGIBILITY = ("s2n", "n_udyam")
_ALWAYS_WITH = {
    "interest": ("s15", "s16", "n_rate") + _ELIGIBILITY,
    "late": ("s15", "s16", "n_rate") + _ELIGIBILITY,
    "complaint": ("s18", "n_samadhaan", "s15", "s16", "s19") + _ELIGIBILITY,
    "tax": ("s23", "n_43bh"),
    "register": ("s2n", "s8", "n_udyam", "n_size"),
    "eligible": ("s2n", "s8", "n_udyam", "n_size"),
    "appeal": ("s19",),
    "deadline": ("s15", "s2b"),
}
_CUSTOMER_QUESTION = ("s15", "s16", "n_rate", "s2b", "s2n", "n_udyam", "s18", "n_samadhaan", "s19", "s23", "n_43bh")


def retrieve(question: str, named_customers: Sequence[str], k: int = 4) -> List[Passage]:
    """BM25 hits for the question plus the passages its topic always needs."""
    hits = [p for p, _ in _retriever.search(question, k=k)]
    needed = list(_CUSTOMER_QUESTION) if named_customers else []
    for tag in sorted(concepts(question)):
        needed += _ALWAYS_WITH.get(tag, ())
    if not hits and not needed and re.search(r"[^\x00-\x7F]", question):
        # A Hindi / Telugu question the keyword list missed: give the model the core late-payment sections.
        needed = list(_CUSTOMER_QUESTION)
    for pid in needed:
        p = PASSAGE_BY_ID[pid]
        if p not in hits:
            hits.append(p)
    return hits


# --------------------------------------------------------------------------- #
# Step 3a: offline answer (no network)
# --------------------------------------------------------------------------- #
def _eligibility_line(f: Facts) -> str:
    if f.size_class == "medium or larger":
        return ("Your turnover is above the small-enterprise limit, so the Act's late-payment protection "
                "may not apply [Sec 2(n)].")
    return (f"This applies **if your business is registered on Udyam as a micro or small enterprise** "
            f"[Sec 2(n), Udyam registration]. By turnover you are a {f.size_class} enterprise.")


def _interest_table(lines: Sequence[InterestLine]) -> str:
    rows = ["| Invoice | Amount | Legal due date | Days late | Interest so far |", "|---|---|---|---|---|"]
    for x in lines:
        rows.append(f"| {x.invoice_id} | {inr(x.amount)} | {_d(x.legal_due_date)} | "
                    f"{x.days_late if x.days_late else 'not yet'} | {inr(x.interest) if x.days_late else '–'} |")
    return "\n".join(rows)


_STEPS = (
    "**What to do next:**\n"
    "1. Send a written demand quoting Sections 15 and 16 and the interest owed.\n"
    "2. If it stays unpaid, file on the MSME Samadhaan portal. The state Facilitation Council must decide "
    "within 90 days [Sec 18, MSME Samadhaan].\n"
    "3. If the buyer challenges the award in court, they must first deposit 75% of it [Sec 19]."
)
_LEVERAGE = ("**Extra leverage:** the buyer cannot deduct this interest from their taxable income [Sec 23], and "
             "while your bill is past 45 days they also lose the tax deduction for the bill itself "
             "[Income-tax Sec 43B(h)].")


def _customer_answer(name: str, f: Facts) -> str:
    cl = f.lines_by_customer.get(name, [])
    if not cl:
        return f"**{name}** has no unpaid invoices in your ledger, so there is nothing to charge interest on."
    late = [x for x in cl if x.days_late > 0]
    parts = []
    if late:
        parts.append(f"**Yes. {name} owes you interest on {len(late)} late invoice{'s' if len(late) > 1 else ''}: "
                     f"{inr(sum(x.interest for x in late))} so far.** {_eligibility_line(f)}")
    else:
        first_due = min(x.legal_due_date for x in cl)
        parts.append(f"**Not yet.** None of {name}'s unpaid invoices is past its legal due date. Interest starts "
                     f"the day after {_d(first_due)} if it is still unpaid then [Sec 16]. {_eligibility_line(f)}")
    parts.append(f"**The rule:** a buyer must pay by the agreed date, and the agreed period can never exceed "
                 f"{MAX_CREDIT_DAYS} days [Sec 15]. Paying late costs compound interest at 3 x the RBI bank rate, "
                 f"{MSMED_RATE:.2%} a year today, and no contract can remove it [Sec 16, RBI Bank Rate].")
    parts.append(_interest_table(cl))
    over = [x for x in cl if x.terms_over_limit and x.days_late > 0]
    if over:
        ids = ", ".join(x.invoice_id for x in over)
        parts.append(f"**Note:** {ids} {'has' if len(over) == 1 else 'have'} {over[0].agreed_terms_days}-day terms, "
                     f"but the law caps credit at {MAX_CREDIT_DAYS} days, so interest started on day "
                     f"{MAX_CREDIT_DAYS + 1} [Sec 15].")
    disputed = [x for x in cl if x.disputed]
    if disputed:
        ids = ", ".join(x.invoice_id for x in disputed)
        parts.append(f"**Check first:** {ids} is marked disputed. If {name} objected in writing within 15 days of "
                     f"delivery, interest starts only from the day you resolved that objection [Sec 2(b)].")
    if late:
        parts += [_STEPS, _LEVERAGE]
    return "\n\n".join(parts)


def offline_answer(question: str, passages: Sequence[Passage], f: Facts) -> str:
    tags = concepts(question)
    if f.customers:
        return "\n\n---\n\n".join(_customer_answer(n, f) for n in f.customers)
    if "complaint" in tags:
        return ("**You can take an unpaid bill to your state's MSME Facilitation Council.** It first tries "
                "conciliation, then arbitration, and must decide within 90 days. It can hear a buyer located "
                "anywhere in India [Sec 18].\n\n"
                "1. Send the buyer a written demand quoting Sections 15 and 16 and the interest owed.\n"
                "2. If it stays unpaid, file online on the MSME Samadhaan portal with your Udyam number and the "
                "unpaid invoices [MSME Samadhaan].\n"
                "3. If the buyer challenges the award in court, they must first deposit 75% of it [Sec 19].\n\n"
                + _eligibility_line(f))
    if "tax" in tags:
        return ("**Yes, paying you late costs the buyer tax.**\n\n- The Section 16 interest they pay you is not "
                "deductible from their income [Sec 23].\n- If they pay after the 45-day limit, they can deduct the "
                "bill only in the year they actually pay it. A bill still unpaid at 31 March counts as profit "
                "for them that year [Income-tax Sec 43B(h)].\n\nMention this to slow payers before the financial "
                "year ends.")
    if "appeal" in tags:
        return ("**A buyer who wants to challenge the Council's award in court must first deposit 75% of the "
                "amount**, and the court can release part of it to you while the case runs [Sec 19].")
    if tags & {"register", "eligible"}:
        return (f"**Only registered micro and small enterprises are protected** [Sec 2(n)]. Registration is the "
                f"free Udyam registration on udyamregistration.gov.in [Sec 8(1), Udyam registration].\n\n"
                f"Size limits from 1 April 2025: micro up to ₹10 Cr turnover, small up to ₹100 Cr turnover "
                f"(and ₹25 Cr investment) [MSME size limits]. With {inr(f.turnover)} turnover you are a "
                f"**{f.size_class} enterprise**.")
    if tags & {"interest", "late"}:
        late = [x for x in f.all_lines if x.days_late > 0]
        total = sum(x.interest for x in late)
        return (f"**Yes. Any buyer who pays after the legal due date owes you compound interest at 3 x the RBI bank "
                f"rate: {MSMED_RATE:.2%} a year today** [Sec 15, Sec 16, RBI Bank Rate]. {_eligibility_line(f)}\n\n"
                f"Across your ledger, {len(late)} unpaid invoices are past the legal due date, with "
                f"**{inr(total)}** of interest so far. The largest:\n\n{_interest_table(late[:5])}\n\n"
                f"Ask about a customer by name (for example *Can I charge Vindhya interest?*) for their full "
                f"breakdown.")
    if "deadline" in tags:
        n_long = sum(1 for x in f.all_lines if x.terms_over_limit)
        return (f"**A buyer must pay within the agreed credit period, and that period can never exceed "
                f"{MAX_CREDIT_DAYS} days from delivery** [Sec 15]. Without a written agreement the limit is 15 days "
                f"[Sec 2(b)].\n\n{n_long} of your unpaid invoices have longer terms than the law allows; for those, "
                f"interest legally starts on day {MAX_CREDIT_DAYS + 1}.")
    if passages:
        top = passages[:2]
        return ("Here is what the Act says:\n\n" + "\n\n".join(f"- **{p.title}** [{p.cite}]: {p.plain}" for p in top))
    return ("I could not find this in the MSMED Act sections I have. I can answer questions about late payments, "
            "interest, the 45-day limit, registration, complaints and the buyer's tax position.")


# --------------------------------------------------------------------------- #
# Step 3b: Claude (online)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are the legal-information assistant inside Runway, a cash-flow early-warning app for \
Indian MSME manufacturers. The person asking owns the business described in <business_facts>. They are not a \
lawyer or a finance expert, so use plain words.

Answer using only the passages in <sources> and the figures in <business_facts>.
- After each legal point, cite the passage label in square brackets, for example [Sec 16] or \
[Income-tax Sec 43B(h)].
- Copy rupee figures and dates from <business_facts> exactly; do not recompute them.
- Explain the law in your own plain words; do not copy more than a few words of the sources.
- Mention any condition that changes the answer, such as Udyam registration or a written objection by the buyer.
- If the sources do not cover the question, say so in one sentence and suggest the state MSME Facilitation \
Council or a lawyer. Do not answer from memory.
- Reply in the language and script the question is written in (English, Hindi, Telugu or Hinglish).
- Start with a one-sentence answer in bold, then at most about 150 words of short bullet points, then one \
concrete next step."""


def _user_block(question: str, passages: Sequence[Passage], f: Facts) -> str:
    src = "\n".join(
        f'<source label="{p.cite}" kind="{"exact text of the MSMED Act 2006" if p.kind == "law" else "Runway note"}" '
        f'title="{p.title}">\n{p.text}\n</source>' for p in passages)
    return f"<sources>\n{src}\n</sources>\n\n<business_facts>\n{f.text}\n</business_facts>\n\n<question>{question}</question>"


def stream_claude_answer(question: str, passages: Sequence[Passage], f: Facts, history: Sequence[dict],
                         api_key: str) -> Iterator[str]:
    """Yield the answer text as it streams. Raises anthropic errors; the caller falls back to offline."""
    import anthropic  # imported here so the app still runs where the SDK is not installed

    # No retries and a short connect timeout: on a flaky venue network, fall back to the offline answer
    # in seconds instead of making the room wait.
    client = anthropic.Anthropic(api_key=api_key, max_retries=0,
                                 timeout=anthropic.Timeout(60.0, connect=4.0))
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    messages.append({"role": "user", "content": _user_block(question, passages, f)})
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=messages,
        output_config={"effort": "low"},          # short chat answers: keep latency low on stage
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                        # if Opus declines, the API retries on a fallback model
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_Claude declined to answer this question._"
    elif final.stop_reason == "max_tokens":
        yield " …"


class GeminiError(Exception):
    """HTTP error from the Gemini API (bad key, quota, unknown model, ...)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code, self.message = status_code, message


def gemini_label(model: str) -> str:
    """'gemini-3.6-flash' -> 'Gemini 3.6 Flash'."""
    return "Gemini " + " ".join(p if p[:1].isdigit() else p.capitalize() for p in model.split("-")[1:])


def stream_gemini_answer(question: str, passages: Sequence[Passage], f: Facts, history: Sequence[dict],
                         api_key: str, model: str = GEMINI_DEFAULT_MODEL) -> Iterator[str]:
    """Same grounding and instructions as Claude, sent to Google's Gemini REST API.

    Plain HTTPS through `requests` (already a Streamlit dependency), so no extra compiled packages.
    Raises GeminiError on an HTTP error and requests.RequestException on a network failure.
    """
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in history[-6:]]
    contents.append({"role": "user", "parts": [{"text": _user_block(question, passages, f)}]})
    body = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents}
    resp = requests.post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": api_key}, json=body,
                         stream=True, timeout=(4, 60))
    with resp:
        if resp.status_code != 200:
            try:
                message = resp.json().get("error", {}).get("message", "")
            except ValueError:
                message = resp.text[:200]
            raise GeminiError(resp.status_code, message)
        resp.encoding = "utf-8"            # keeps Hindi / Telugu intact; SSE responses carry no charset
        finish = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:].strip())
            if chunk.get("promptFeedback", {}).get("blockReason"):
                yield "\n\n_Gemini declined to answer this question._"
                return
            for cand in chunk.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        yield part["text"]
                finish = cand.get("finishReason") or finish
    if finish != "STOP":
        # Cut short (MAX_TOKENS, RECITATION, SAFETY, dropped connection ...): let the caller add the offline answer.
        raise GeminiError(0, f"answer stopped early ({finish or 'connection ended'})")


def answer_offline(question: str, ctx: BusinessContext) -> tuple:
    f = build_facts(question, ctx)
    passages = retrieve(question, f.customers)
    return offline_answer(question, passages, f), passages, f


def prepare(question: str, ctx: BusinessContext) -> tuple:
    """Retrieval + grounding, shared by both generation paths."""
    f = build_facts(question, ctx)
    return retrieve(question, f.customers), f


def cited_passages(answer: str, passages: Sequence[Passage]) -> List[Passage]:
    """Passages whose label appears in the answer, else all retrieved ones."""
    used = [p for p in passages if p.cite in answer]
    return used or list(passages)


__all__ = ["BusinessContext", "Facts", "CLAUDE_LABEL", "CLAUDE_MODEL", "GEMINI_DEFAULT_MODEL", "GeminiError",
           "answer_offline", "prepare", "offline_answer", "stream_claude_answer", "stream_gemini_answer",
           "gemini_label", "cited_passages", "inr"]
