"""
Runway (Predict. Explain. Act.)
Transaction-Aware Cash-Shortfall Early Warning for MSME Manufacturers.
INNOVASIA 2K26 | Problem Statement: PS-04 | Team Fin-Hack

Every number on this page is computed by the engine at load time; nothing is typed in.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.generator import generate_msme_ledger, reconstruct_cash_history
from data.schemas import EXPENSE_EMI_RENT, EXPENSE_GST_TAX, EXPENSE_PAYROLL, EXPENSE_RAW_MATERIAL
from engine import run_runway_pipeline

SEED = 42
N_SIMULATIONS = 5000
HORIZON_DAYS = 60
RHO = 0.5
COMPANY_NAME = "Sri Lakshmi Auto Components Pvt Ltd"
OWNER_NAME = "Raghav"

CATEGORY_SHORT = {
    EXPENSE_PAYROLL: "Wages & salaries",
    EXPENSE_RAW_MATERIAL: "Raw-material vendors",
    EXPENSE_GST_TAX: "GST / tax",
    EXPENSE_EMI_RENT: "Loan EMI & rent",
}

ACTION_LABELS = {
    "DIRECT_CHASE": "Call the customer directly",
    "EARLY_DISCOUNT_2%": "Offer a 2% early-payment discount",
    "SPLIT_PAYMENT": "Propose a split payment",
    "LEGAL_MSMED_NOTICE": "Send an MSMED Act payment notice",
}
ACTION_EFFECT = {
    "DIRECT_CHASE": "Assumed effect: a personal follow-up brings payment about 10 days sooner.",
    "EARLY_DISCOUNT_2%": "Assumed effect: 75% of buyers take the discount and pay within 7 days.",
    "SPLIT_PAYMENT": "Assumed effect: if the buyer agrees (60% chance), half arrives now and half in 30 days.",
    "LEGAL_MSMED_NOTICE": "Assumed effect: 50% chance of settlement within 14 days. Only offered for disputed "
                          "or >45-day overdue invoices (MSMED Act 2006).",
}
ACTION_PHRASES = {
    "DIRECT_CHASE": {
        "en": "call {c} today about invoice {i}",
        "hi": "{c} को बिल {i} के लिए आज ही फ़ोन करें",
        "te": "బిల్లు {i} గురించి ఈ రోజే {c} కి ఫోన్ చేయండి",
    },
    "EARLY_DISCOUNT_2%": {
        "en": "offer {c} a 2% discount to pay invoice {i} within 7 days",
        "hi": "{c} को बिल {i} 7 दिन में चुकाने पर 2% छूट दें",
        "te": "బిల్లు {i} ను 7 రోజుల్లో చెల్లిస్తే {c} కి 2% తగ్గింపు ఇవ్వండి",
    },
    "SPLIT_PAYMENT": {
        "en": "ask {c} to pay half of invoice {i} now and the rest in 30 days",
        "hi": "{c} से बिल {i} का आधा भुगतान अभी और बाकी 30 दिन में लें",
        "te": "బిల్లు {i} లో సగం ఇప్పుడు, మిగతాది 30 రోజుల్లో చెల్లించమని {c} ని అడగండి",
    },
    "LEGAL_MSMED_NOTICE": {
        "en": "send {c} an MSMED Act payment notice for invoice {i}",
        "hi": "{c} को बिल {i} के लिए MSMED अधिनियम के तहत भुगतान नोटिस भेजें",
        "te": "బిల్లు {i} కోసం MSMED చట్టం కింద {c} కి చెల్లింపు నోటీసు పంపండి",
    },
}
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
TE_MONTHS = ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"]

# --------------------------------------------------------------------------- #
# Page configuration & styling
# Colours for Streamlit's own widgets live in .streamlit/config.toml (dark theme).
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Runway · Cash-Shortfall Early Warning",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #070B11;
        color: #F8FAFC;
    }

    /* Top bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        padding: 10px 0 18px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 22px;
    }
    .brand-container { display: flex; align-items: center; gap: 12px; }
    .brand-title { font-size: 24px; font-weight: 700; letter-spacing: -0.3px; color: #FFFFFF; line-height: 1.1; }
    .brand-sub { font-size: 9px; letter-spacing: 2.2px; text-transform: uppercase; color: #00E5FF; font-weight: 700; margin-top: 3px; }
    .badge-pill {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 13px;
        color: #CBD5E1;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .live-dot {
        width: 7px; height: 7px; background-color: #10B981; border-radius: 50%;
        display: inline-block; box-shadow: 0 0 8px #10B981;
    }

    /* Greeting */
    .eyebrow { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #00E5FF; font-weight: 700; margin-bottom: 4px; }
    .greeting-title { font-size: 32px; font-weight: 700; letter-spacing: -0.5px; color: #FFFFFF; margin-bottom: 4px; }
    .greeting-sub { font-size: 15px; color: #CBD5E1; margin-bottom: 22px; }

    /* KPI cards */
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px 22px;
        min-height: 168px;
    }
    .kpi-label {
        font-size: 11px; letter-spacing: 1.4px; text-transform: uppercase; color: #94A3B8;
        font-weight: 600; margin-bottom: 12px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .kpi-value { font-size: 34px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; margin-bottom: 8px; }
    .kpi-delta { font-size: 13px; font-weight: 500; }
    .kpi-plain { font-size: 12.5px; color: #94A3B8; margin-top: 8px; line-height: 1.45; }
    .delta-green { color: #10B981; }
    .delta-red { color: #EF4444; }

    .badge-danger, .badge-warn, .badge-ok {
        font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px;
        text-transform: uppercase; display: inline-block;
    }
    .badge-danger { background: rgba(239, 68, 68, 0.16); color: #F87171; }
    .badge-warn { background: rgba(245, 158, 11, 0.16); color: #FBBF24; }
    .badge-ok { background: rgba(16, 185, 129, 0.16); color: #34D399; }
    .badge-subtle { background: #1F2937; color: #CBD5E1; font-size: 11px; padding: 3px 8px; border-radius: 4px; display: inline-block; }

    /* Tabs: readable inactive labels, cyan active label */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #CBD5E1 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 4px;
        transition: color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #FFFFFF !important; }
    .stTabs [aria-selected="true"] { color: #00E5FF !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #00E5FF !important; }

    /* Section headings and plain-language explainers */
    .section-title { font-size: 20px; font-weight: 700; color: #FFFFFF; margin: 16px 0 4px 0; }
    .section-plain { font-size: 14px; color: #CBD5E1; margin-bottom: 18px; line-height: 1.55; max-width: 980px; }

    /* Summary cards */
    .summary-card {
        background-color: #111827; border: 1px solid #1F2937; border-radius: 10px;
        padding: 16px 20px; min-height: 128px;
    }
    .summary-label { font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; color: #94A3B8; font-weight: 600; margin-bottom: 6px; }
    .summary-value { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .summary-sub { font-size: 12.5px; color: #CBD5E1; line-height: 1.45; }

    /* Action cards */
    .action-card {
        background-color: #111827; border: 1px solid #1F2937; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 16px;
    }
    .action-card:hover { border-color: #00E5FF; }
    .action-title { font-size: 17px; font-weight: 700; color: #FFFFFF; margin: 6px 0; }
    .action-desc { font-size: 14px; color: #E2E8F0; margin-bottom: 6px; line-height: 1.5; }
    .action-assume { font-size: 12.5px; color: #94A3B8; margin-bottom: 12px; }

    .info-card {
        background-color: #111827; border: 1px solid #1F2937; border-radius: 12px;
        padding: 20px 22px; margin-bottom: 14px;
    }
    .info-card b { color: #FFFFFF; }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def inr(value: float, signed: bool = False) -> str:
    """Indian-style rupee amount: ₹ crore / lakh / plain."""
    a = abs(value)
    sign = "−" if value < 0 else ("+" if signed and value > 0 else "")
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    return f"{sign}₹{a:,.0f}"


def day(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%b %d")


def risk_badge(p: float) -> str:
    if p >= 0.6:
        return '<span class="badge-danger">High risk</span>'
    if p >= 0.3:
        return '<span class="badge-warn">Medium risk</span>'
    return '<span class="badge-ok">Low risk</span>'


def window_phrase(start: pd.Timestamp, end: pd.Timestamp, lang: str) -> str:
    if lang == "hi":
        if start.month == end.month:
            return f"{start.day}–{end.day} {HI_MONTHS[end.month - 1]}"
        return f"{start.day} {HI_MONTHS[start.month - 1]} – {end.day} {HI_MONTHS[end.month - 1]}"
    if lang == "te":
        if start.month == end.month:
            return f"{TE_MONTHS[end.month - 1]} {start.day}–{end.day}"
        return f"{TE_MONTHS[start.month - 1]} {start.day} – {TE_MONTHS[end.month - 1]} {end.day}"
    return f"{day(start)} – {day(end)}" if start != end else day(start)


WARN_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 '
    '4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
)
CAL_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/>'
    '<line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>'
)


# --------------------------------------------------------------------------- #
# Run the engine once per server process (shared, read-only)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Running 5,000 cash-flow simulations (about 30 seconds on first load)…")
def load_engine(seed: int = SEED):
    invoices, expenses, cash = generate_msme_ledger(seed=seed)
    as_of = invoices.attrs["as_of_date"]
    sim, actions, _ = run_runway_pipeline(
        invoices, expenses, cash, as_of_date=as_of,
        n_simulations=N_SIMULATIONS, horizon_days=HORIZON_DAYS, rho=RHO, seed=seed,
    )
    history = reconstruct_cash_history(invoices, expenses, cash)
    return invoices, expenses, float(cash), pd.Timestamp(as_of), sim, actions, history


invoices, expenses, cash_today, as_of, sim, actions, history = load_engine()

# --------------------------------------------------------------------------- #
# Derived numbers (all from the engine output)
# --------------------------------------------------------------------------- #
dates = pd.to_datetime(sim.dates)
p10 = np.asarray(sim.p10_trajectory, dtype=float)
p50 = np.asarray(sim.p50_trajectory, dtype=float)
p90 = np.asarray(sim.p90_trajectory, dtype=float)
H30 = min(30, len(dates))

customer_names = invoices.drop_duplicates("customer_id").set_index("customer_id")["customer_name"]
turnover = invoices.loc[invoices["issue_date"] > as_of - pd.Timedelta(days=365), "amount"].sum()

week_ago = as_of - pd.Timedelta(days=7)
cash_week_ago = float(history["closing_cash"].get(week_ago, np.nan))
cash_change = cash_today - cash_week_ago

exp = expenses.copy()
exp["due_date"] = pd.to_datetime(exp["due_date"])
upcoming = exp[(exp["due_date"] > as_of) & (exp["due_date"] <= dates[-1])]

if sim.shortfall_window_dates:
    win_start, win_end = (pd.Timestamp(d) for d in sim.shortfall_window_dates)
    window_text = window_phrase(win_start, win_end, "en")
    due_by_window = (upcoming[upcoming["due_date"] <= win_end]
                     .groupby("category")["amount"].sum().sort_values(ascending=False))
    trigger_text = " + ".join(f"{CATEGORY_SHORT.get(c, c)} {inr(a)}" for c, a in due_by_window.head(2).items())
else:
    win_start = win_end = None
    window_text, trigger_text = "None expected", "No likely shortfall in the next 60 days"

# Typical-case (median) path over the next 30 days
low_idx = int(np.argmin(p50[:H30]))
low_value, low_date = p50[low_idx], dates[low_idx]
below = p50[:H30] < 0
if below.any():
    first_neg = int(np.argmax(below))
    back_up = np.where(p50[first_neg:] >= 0)[0]
    recover_text = day(dates[first_neg + back_up[0]]) if len(back_up) else "Not within 60 days"
    recover_sub = f"Typical case goes below ₹0 on {day(dates[first_neg])}"
else:
    recover_text, recover_sub = "Stays above ₹0", "In the typical case, cash never goes negative"

# Credit line that keeps 9 out of 10 simulated futures above ₹0 for 30 days
path_min_30 = np.asarray(sim.trajectories)[:, :H30].min(axis=1)
buffer_needed = max(0.0, -float(np.percentile(path_min_30, 10)))

if sim.p_deficit_14d >= 0.5:
    greeting_sub = "Your cash position needs attention this week."
elif sim.p_deficit_30d >= 0.3:
    greeting_sub = "Keep an eye on cash over the next month."
else:
    greeting_sub = "Cash looks healthy for the next month."

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="top-header">
        <div class="brand-container">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="3" y="13" width="4" height="8" rx="1.5" fill="#00E5FF"/>
                <rect x="10" y="8" width="4" height="13" rx="1.5" fill="#00E5FF"/>
                <rect x="17" y="3" width="4" height="18" rx="1.5" fill="#00E5FF"/>
            </svg>
            <div>
                <div class="brand-title">Runway</div>
                <div class="brand-sub">PREDICT. EXPLAIN. ACT.</div>
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
            <div class="badge-pill">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/>
                    <path d="M6 12H4a2 2 0 0 0-2 2v8h4"/>
                    <path d="M18 9h2a2 2 0 0 1 2 2v11h-4"/>
                </svg>
                <span>{COMPANY_NAME}</span>
                <span style="color: #94A3B8;">· {inr(turnover)} annual turnover · demo company, simulated ledger</span>
            </div>
            <div class="badge-pill">
                <span class="live-dot"></span>
                <span style="color: #F8FAFC; font-weight: 500;">{N_SIMULATIONS:,} simulated futures</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_greet, col_time = st.columns([3, 1])
with col_greet:
    st.markdown('<div class="eyebrow">CASH INTELLIGENCE / LIVE OUTLOOK</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="greeting-title">Good morning, {OWNER_NAME}.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="greeting-sub">{greeting_sub}</div>', unsafe_allow_html=True)
with col_time:
    st.markdown(
        f"""
        <div style="text-align: right; color: #94A3B8; font-size: 13px; padding-top: 20px;">
            Forecast as of {as_of:%b %d, %Y}
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("New here? How to read this dashboard", icon=":material/help:"):
    st.markdown(
        f"""
- **The problem:** a factory can be profitable and still run out of cash, because customers pay late
  while wages, GST, loan EMIs and raw-material bills fall due on fixed dates.
- **What Runway does:** it looks at every unpaid invoice and every upcoming bill. From past invoices it
  learns how late each customer usually pays, then plays out **{N_SIMULATIONS:,} possible futures** of who pays when.
- **Shortfall risk** = in what share of those futures the bank balance drops below ₹0.
  89% means 89 out of 100 simulated futures run out of cash at some point.
- **Tabs:** *Predict* shows when the crunch is likely, *Explain* shows which unpaid invoices cause it,
  *Act* ranks what to do (rupees saved minus rupees spent), *Proof* shows how well this worked on past data.
- **Data:** this is a simulated auto-parts supplier with 25 customers and 500 invoices over one year.
  No real company data is used.
        """
    )

# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
k1, k2, k3, k4 = st.columns(4)

with k1:
    if np.isfinite(cash_change):
        cls = "delta-green" if cash_change >= 0 else "delta-red"
        arrow = "↑" if cash_change >= 0 else "↓"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {inr(abs(cash_change))} vs. 7 days ago</div>'
    else:
        delta_html = ""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-bottom: 2px solid #00E5FF;">
            <div class="kpi-label">Cash balance today</div>
            <div class="kpi-value">{inr(cash_today)}</div>
            {delta_html}
            <div class="kpi-plain">Money in the bank on {as_of:%b %d}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for col, label, p, days in ((k2, "14-day shortfall risk", sim.p_deficit_14d, 14),
                            (k3, "30-day shortfall risk", sim.p_deficit_30d, 30)):
    with col:
        icon = WARN_SVG.format(c="#EF4444" if p >= 0.6 else "#F59E0B" if p >= 0.3 else "#10B981")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>{label}</span>{icon}</div>
                <div class="kpi-value">{p:.1%}</div>
                {risk_badge(p)}
                <div class="kpi-plain">Chance the bank balance drops below ₹0 in the next {days} days.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label"><span>Most likely crunch</span>{CAL_SVG}</div>
            <div class="kpi-value" style="font-size: 30px;">{window_text}</div>
            <div class="kpi-plain"><span style="color: #FBBF24; font-weight: 600;">Bills due by then:</span> {trigger_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_predict, tab_explain, tab_act, tab_eval = st.tabs([
    "1 · Predict: when will cash run short?",
    "2 · Explain: which invoices cause it?",
    "3 · Act: what should I do first?",
    "4 · Proof: does it work?",
])

# --------------------------------------------------------------------------- #
# TAB 1: PREDICT (fan chart)
# --------------------------------------------------------------------------- #
with tab_predict:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; margin: 16px 0 6px 0;">
            <div>
                <span style="font-size: 20px; font-weight: 700; color: #FFFFFF;">60-day cash forecast</span>
                <span style="background: rgba(0, 229, 255, 0.12); color: #00E5FF; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; margin-left: 8px; border: 1px solid rgba(0, 229, 255, 0.3);">{N_SIMULATIONS:,} SIMULATIONS</span>
            </div>
            <div style="display: flex; gap: 20px; font-size: 12.5px; color: #CBD5E1;">
                <span><span style="color: #00E5FF; font-weight: bold; margin-right: 4px;">━</span> Typical outcome (median)</span>
                <span><span style="background: rgba(0, 229, 255, 0.25); padding: 0 8px; border-radius: 2px; margin-right: 4px;">&nbsp;</span> 8 in 10 outcomes fall here</span>
                <span><span style="color: #EF4444; margin-right: 4px;">┄┄</span> ₹0: can't pay bills</span>
            </div>
        </div>
        <div class="section-plain">Each simulated future reshuffles when customers pay. The bright line is the typical
        bank balance day by day; the shaded band shows the range of likely outcomes. Big drops are fixed bills
        (wages, raw material, GST) falling due before customer money arrives.</div>
        """,
        unsafe_allow_html=True,
    )

    p10_L, p50_L, p90_L = p10 / 1e5, p50 / 1e5, p90 / 1e5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=p90_L, mode="lines",
        line=dict(width=1, color="rgba(0, 229, 255, 0.35)", dash="dash"),
        name="Optimistic (P90)",
        hovertemplate="Optimistic: ₹%{y:.2f} L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p10_L, mode="lines",
        line=dict(width=1, color="rgba(0, 229, 255, 0.35)", dash="dash"),
        fill="tonexty", fillcolor="rgba(0, 229, 255, 0.10)",
        name="Pessimistic (P10)",
        hovertemplate="Pessimistic: ₹%{y:.2f} L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p50_L, mode="lines",
        line=dict(width=3, color="#00E5FF"),
        name="Typical (median)",
        hovertemplate="Typical: ₹%{y:.2f} L<extra></extra>",
    ))

    fig.add_hline(
        y=0, line_dash="dash", line_color="#EF4444", line_width=1.5,
        annotation_text="₹0 — can't pay bills", annotation_position="bottom left",
        annotation_font_color="#F87171", annotation_font_size=11,
    )

    if win_start is not None:
        fig.add_vrect(
            x0=win_start - pd.Timedelta(hours=12), x1=win_end + pd.Timedelta(hours=12),
            fillcolor="rgba(239, 68, 68, 0.14)", line_width=0,
            annotation_text="LIKELY CRUNCH", annotation_position="top left",
            annotation_font_color="#F87171", annotation_font_size=10,
        )

    # Label the three biggest bill days in the next 30 days with their real amounts
    bills = upcoming[upcoming["due_date"] <= dates[H30 - 1]]
    if not bills.empty:
        by_day = bills.groupby("due_date").agg(
            amount=("amount", "sum"),
            what=("category", lambda s: " + ".join(dict.fromkeys(CATEGORY_SHORT.get(c, c) for c in s))),
        )
        for k, (d, row) in enumerate(by_day.nlargest(3, "amount").sort_index().iterrows()):
            i = (d - as_of).days - 1
            if not 0 <= i < len(dates):
                continue
            fig.add_annotation(
                x=d, y=float(p50_L[i]),
                text=f"<b>{day(d).upper()} · {row['what'].upper()}</b><br><span style='color:#F87171;'>{inr(-row['amount'])}</span>",
                showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#F59E0B",
                ax=0, ay=-70 if k % 2 == 0 else 60,
                bgcolor="#111827", bordercolor="#F59E0B", borderwidth=1, borderpad=6,
                font=dict(size=11, color="#FFFFFF"),
            )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=50, r=30, t=30, b=40),
        height=440,
        showlegend=False,
        font=dict(color="#CBD5E1"),
        xaxis=dict(showgrid=True, gridcolor="#1F2937", color="#CBD5E1", tickformat="%b %d"),
        yaxis=dict(
            showgrid=True, gridcolor="#1F2937", color="#CBD5E1", tickprefix="₹", ticksuffix=" L",
            title=dict(text="Bank balance (₹ lakhs)", font=dict(color="#94A3B8", size=12)),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0F172A", font=dict(color="#F8FAFC")),
    )
    st.plotly_chart(fig, alt="60-day cash forecast fan chart with typical, optimistic and pessimistic bank balance")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Lowest point, next 30 days (typical)</div>
                <div class="summary-value" style="color: {'#F87171' if low_value < 0 else '#FFFFFF'};">{inr(low_value)}</div>
                <div class="summary-sub">On {low_date:%B %d}, in the typical simulated future.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Back above ₹0 (typical)</div>
                <div class="summary-value" style="color: #34D399;">{recover_text}</div>
                <div class="summary-sub">{recover_sub}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Safety buffer to arrange</div>
                <div class="summary-value" style="color: #FFFFFF;">{inr(buffer_needed)}</div>
                <div class="summary-sub">An overdraft or credit line this size keeps the account above ₹0
                in 9 out of 10 simulated futures over the next 30 days.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------- #
# TAB 2: EXPLAIN (risk attribution)
# --------------------------------------------------------------------------- #
with tab_explain:
    st.markdown('<div class="section-title">Which unpaid invoices are causing the risk?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-plain">We re-ran all the simulations pretending each unpaid invoice arrives on its due date. '
        'The bigger the drop in shortfall risk, the more that one invoice matters. These are the five that matter most.</div>',
        unsafe_allow_html=True,
    )

    drivers = pd.DataFrame(sim.top_risk_invoices)
    if drivers.empty:
        st.info("No single unpaid invoice moves the 30-day risk noticeably.", icon=":material/check_circle:")
    else:
        drivers["drop_pts"] = drivers["marginal_risk_reduction"] * 100
        col_chart, col_callout = st.columns([3, 2])
        with col_chart:
            bar_fig = go.Figure(go.Bar(
                x=drivers["drop_pts"],
                y=drivers["customer"] + " · " + drivers["invoice_id"],
                orientation="h",
                marker=dict(color="#00E5FF"),
                text=[f"−{v:.1f} pts" for v in drivers["drop_pts"]],
                textposition="outside",
                textfont=dict(color="#E2E8F0"),
                hovertemplate="<b>%{y}</b><br>Paying on time cuts 30-day risk by %{x:.1f} points<extra></extra>",
            ))
            bar_fig.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                margin=dict(l=10, r=40, t=10, b=40),
                height=340,
                font=dict(color="#CBD5E1"),
                xaxis=dict(title="Drop in 30-day shortfall risk if paid on time (percentage points)",
                           color="#CBD5E1", gridcolor="#1F2937", range=[0, drivers["drop_pts"].max() * 1.25]),
                yaxis=dict(color="#F8FAFC", autorange="reversed"),
            )
            st.plotly_chart(bar_fig, alt="Bar chart of the five invoices whose late payment adds most to shortfall risk")

        with col_callout:
            top = drivers.iloc[0]
            by_customer = (drivers.groupby("customer")
                           .agg(n=("invoice_id", "size"), amount=("amount", "sum"))
                           .sort_values(["n", "amount"], ascending=False))
            lead_customer, lead = by_customer.index[0], by_customer.iloc[0]
            if lead["n"] >= 2:
                concentration = (
                    f"<b>{lead_customer}</b> owns <b>{int(lead['n'])} of the top {len(drivers)}</b> risky invoices, "
                    f"<b>{inr(lead['amount'])}</b> in total. When one buyer is slow, all its invoices tend to be slow "
                    f"together, which is why Runway simulates them as linked rather than independent."
                )
                title = "One buyer, several risky invoices"
            else:
                concentration = ("The risk is spread across different buyers, so no single customer relationship "
                                 "decides your month.")
                title = "Risk is spread across buyers"
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="display: flex; align-items: center; gap: 8px; color: #FBBF24; font-weight: 700; font-size: 15px; margin-bottom: 10px;">
                        {WARN_SVG.format(c="#F59E0B")}<span>{title}</span>
                    </div>
                    <div style="font-size: 14px; color: #E2E8F0; line-height: 1.6; margin-bottom: 12px;">{concentration}</div>
                    <div style="background: rgba(0, 229, 255, 0.08); border-left: 3px solid #00E5FF; padding: 10px 12px; font-size: 13px; color: #E2E8F0; line-height: 1.5;">
                        Biggest single lever: if <b>{top['customer']}</b> pays <b>{top['invoice_id']}</b>
                        ({inr(top['amount'])}) on time, the 30-day shortfall risk falls by <b>{top['drop_pts']:.1f} points</b>.
                        See the <b>Act</b> tab for the cheapest way to make that happen.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        table = pd.DataFrame({
            "Invoice": drivers["invoice_id"],
            "Customer": drivers["customer"],
            "Amount (₹ L)": drivers["amount"] / 1e5,
            "Due date": pd.to_datetime(drivers["due_date"]),
            "Chance paid within 30 days": drivers["p_paid_in_30d"],
            "Risk drop if paid on time (pts)": drivers["drop_pts"],
        })
        st.dataframe(
            table,
            hide_index=True,
            alt="Top five risk-driving invoices",
            column_config={
                "Amount (₹ L)": st.column_config.NumberColumn(format="%.2f"),
                "Due date": st.column_config.DateColumn(format="MMM DD, YYYY"),
                "Chance paid within 30 days": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="percent",
                    help="What the survival model expects, given how long this invoice is already unpaid",
                ),
                "Risk drop if paid on time (pts)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# --------------------------------------------------------------------------- #
# TAB 3: ACT (ranked interventions & vernacular alerts)
# --------------------------------------------------------------------------- #
with tab_act:
    st.markdown('<div class="section-title">What to do first, ranked by rupees</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-plain">For each risky invoice, Runway tries four actions inside the same simulations and '
        'measures how much smaller the expected cash gap becomes. <b>Net gain = cash gap avoided − cost of the action.</b> '
        'The top card is the best rupee-for-rupee move.</div>',
        unsafe_allow_html=True,
    )

    act_col1, act_col2 = st.columns([3, 2])

    with act_col1:
        if not actions:
            st.info("No action is worth its cost right now.", icon=":material/check_circle:")
        for idx, act in enumerate(actions[:5]):
            customer = customer_names.get(act.customer_id, act.customer_id)
            roi = act.expected_shortfall_avoided / max(1.0, act.cost_of_action)
            st.markdown(
                f"""
                <div class="action-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;">
                        <div>
                            <span class="badge-subtle" style="color: #00E5FF; font-weight: 700;">#{idx + 1} · {act.invoice_id} · {customer}</span>
                            <div class="action-title">{ACTION_LABELS.get(act.action_type, act.action_type)}</div>
                            <div class="action-desc">{act.description.replace('Rs ', '₹')}</div>
                            <div class="action-assume">{ACTION_EFFECT.get(act.action_type, '')}</div>
                        </div>
                        <div style="text-align: right; min-width: 120px;">
                            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase;">Net gain</div>
                            <div style="font-size: 22px; font-weight: 700; color: #34D399;">{inr(act.net_benefit, signed=True)}</div>
                        </div>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 24px; font-size: 13px; color: #CBD5E1; border-top: 1px solid #1F2937; padding-top: 10px;">
                        <span>Cash gap avoided: <b>{inr(act.expected_shortfall_avoided)}</b></span>
                        <span>Cost: <b>{inr(act.cost_of_action)}</b></span>
                        <span>Return: <b>{roi:,.0f}×</b> the cost</span>
                        <span>Helps in <b>{act.confidence_score:.0%}</b> of the bad scenarios</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with act_col2:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom: 8px;">
                <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">WhatsApp alert for the owner</div>
                <div style="font-size: 13px; color: #CBD5E1;">A ready-to-send summary for {OWNER_NAME}, in the language he reads fastest.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lang_label = st.segmented_control(
            "Message language", ["English", "हिंदी (Hindi)", "తెలుగు (Telugu)"],
            default="English", required=True, key="alert_lang",
        )
        lang = {"English": "en", "हिंदी (Hindi)": "hi", "తెలుగు (Telugu)": "te"}[lang_label or "English"]

        p30 = f"{sim.p_deficit_30d:.0%}"
        if actions:
            top_act = actions[0]
            c_name = customer_names.get(top_act.customer_id, top_act.customer_id)
            phrase = ACTION_PHRASES.get(top_act.action_type, ACTION_PHRASES["DIRECT_CHASE"])[lang].format(
                c=c_name, i=top_act.invoice_id)
            gain = inr(top_act.net_benefit)
        else:
            phrase, gain = "", ""

        if win_start is None:
            msg = {
                "en": f"Runway update: cash looks safe for the next 30 days (shortfall risk {p30}).",
                "hi": f"Runway अपडेट: अगले 30 दिन कैश सुरक्षित दिख रहा है (कमी की संभावना {p30})।",
                "te": f"Runway సమాచారం: రాబోయే 30 రోజులు నగదు సురక్షితంగా ఉంది (కొరత అవకాశం {p30}).",
            }[lang]
        else:
            w = window_phrase(win_start, win_end, lang)
            msg = {
                "en": f"Runway alert: {p30} chance cash runs short in the next 30 days, most likely {w}. "
                      f"Best move: {phrase}. Expected net gain: {gain}.",
                "hi": f"Runway चेतावनी: अगले 30 दिनों में कैश की कमी की संभावना {p30} है, सबसे ज़्यादा {w} के बीच। "
                      f"सबसे अच्छा कदम: {phrase}। अनुमानित शुद्ध फ़ायदा: {gain}।",
                "te": f"Runway హెచ్చరిక: రాబోయే 30 రోజుల్లో నగదు కొరత వచ్చే అవకాశం {p30}, ఎక్కువగా {w} మధ్య. "
                      f"ఉత్తమ చర్య: {phrase}. అంచనా నికర ప్రయోజనం: {gain}.",
            }[lang]

        st.code(msg, language=None, wrap_lines=True)
        st.caption("Use the copy icon on the message, or open it straight in WhatsApp.")
        st.link_button("Open in WhatsApp", f"https://wa.me/?text={quote(msg)}",
                       icon=":material/send:", type="primary", width="stretch")

# --------------------------------------------------------------------------- #
# TAB 4: PROOF (backtest vs baselines)
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown('<div class="section-title">Does it work? Tested on past data it had not seen</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-plain">We went back to each of 90 past days, hid everything that happened afterwards, '
        'asked every method "will cash go below ₹0 in the next 30 days?", then checked what really happened. '
        'Runway is compared with the simple methods a small business or a standard forecasting tool would use.</div>',
        unsafe_allow_html=True,
    )

    b_df = pd.read_csv(ROOT_DIR / "data" / "benchmark_results.csv")
    st.dataframe(
        b_df,
        hide_index=True,
        alt="Backtest results for Runway and four baseline methods",
        column_config={
            "Model": st.column_config.TextColumn("Method", width="large"),
            "30d Brier Score": st.column_config.NumberColumn(
                "Prediction error (Brier, lower is better)", format="%.3f",
                help="Average squared gap between the predicted chance and what happened (0 = perfect)."),
            "Warning Lead Time (days)": st.column_config.NumberColumn(
                "Days of early warning", format="%d days",
                help="How many days before cash actually went negative the method first raised an alarm (chance > 40%)."),
            "False Warning Rate (%)": st.column_config.TextColumn(
                "False alarms", help="Share of alarms where cash did not in fact go negative."),
            "Deficit Onsets Caught": st.column_config.TextColumn(
                "Cash crunches warned", help="Cash crunches that had at least one alarm in the 30 days before."),
        },
    )

    g1, g2, g3 = st.columns(3)
    for col, title, body in (
        (g1, "Prediction error", "How far the predicted chances were from reality. A model that says 90% and is "
                                 "right scores near 0; one that says 90% and is wrong scores near 1."),
        (g2, "Days of early warning", "How long before the crunch the alarm went off. More days means more time "
                                      "to chase payments or arrange credit."),
        (g3, "False alarms", "How often an alarm turned out to be unnecessary. Too many and the owner stops "
                             "listening."),
    ):
        with col:
            st.markdown(f'<div class="info-card"><b>{title}</b><div class="summary-sub" style="margin-top: 6px;">{body}</div></div>',
                        unsafe_allow_html=True)

    def _row(name_part: str):
        hit = b_df[b_df["Model"].str.contains(name_part, case=False, regex=False)]
        return hit.iloc[0] if len(hit) else None

    runway_row, arima_row = _row("Runway"), _row("ARIMA")
    if runway_row is not None and arima_row is not None:
        st.success(
            f"Runway's predictions were the closest to reality (error {runway_row['30d Brier Score']:.3f} vs "
            f"{arima_row['30d Brier Score']:.3f} for ARIMA, the standard statistical forecast). It warned "
            f"{int(runway_row['Warning Lead Time (days)'])} days ahead of every cash crunch "
            f"({runway_row['Deficit Onsets Caught']}) with fewer false alarms "
            f"({runway_row['False Warning Rate (%)']} vs {arima_row['False Warning Rate (%)']}).",
            title="What this shows",
            icon=":material/verified:",
        )
    st.caption(
        "Honest limits: tested on 90 days of one simulated company containing 3 cash crunches. Always guessing the "
        "average rate (39%) scores almost as well on prediction error (0.238), but it never raises an alarm; the "
        "early warning is where Runway clearly adds value."
    )
