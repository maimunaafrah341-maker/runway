"""
Data contracts and schema definitions for Runway.
Both backend modeling and frontend UI rely strictly on these structures.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# Canonical categorical values. Import these instead of re-typing the strings.
SEGMENT_TIER1_OEM = "Tier-1 Auto OEM"
SEGMENT_MID_TIER = "Mid-tier Supplier"
SEGMENT_RETAIL = "Retail Distributor"
CUSTOMER_SEGMENTS = (SEGMENT_TIER1_OEM, SEGMENT_MID_TIER, SEGMENT_RETAIL)

EXPENSE_PAYROLL = "Payroll"
EXPENSE_GST_TAX = "GST/Tax (GSTR-3B)"
EXPENSE_RAW_MATERIAL = "Raw Material"
EXPENSE_EMI_RENT = "EMI/Rent"
EXPENSE_CATEGORIES = (EXPENSE_PAYROLL, EXPENSE_GST_TAX, EXPENSE_RAW_MATERIAL, EXPENSE_EMI_RENT)

@dataclass
class InvoiceRecord:
    invoice_id: str
    customer_id: str
    customer_segment: str      # One of CUSTOMER_SEGMENTS
    amount: float              # In INR (₹)
    issue_date: str            # YYYY-MM-DD
    due_date: str              # YYYY-MM-DD
    actual_payment_date: Optional[str]  # None if unpaid (right-censored)
    payment_terms_days: int    # 30, 45 (MSMED Act limit), or 60
    is_disputed: bool
    customer_past_avg_delay: float

@dataclass
class RecurringExpense:
    expense_id: str
    category: str              # One of EXPENSE_CATEGORIES
    amount: float
    due_date: str              # YYYY-MM-DD
    is_flexible: bool          # Can payment be delayed by 7 days without shutdown?

@dataclass
class SimulationResult:
    trajectories: Any          # np.ndarray (n_simulations, horizon_days): closing cash per path per day
    dates: List[str]           # YYYY-MM-DD for each trajectory column (day 1 = day after as-of)
    p_deficit_14d: float       # Probability of cash < 0 at any point within 14 days
    p_deficit_30d: float       # Probability within 30 days
    p_deficit_60d: float       # Probability within 60 days
    expected_shortfall_date: Optional[str]  # Median first-deficit date among paths that go negative
    p10_trajectory: List[float]
    p50_trajectory: List[float]
    p90_trajectory: List[float]
    top_risk_invoices: List[dict] # [{invoice_id, customer, amount, marginal_risk_reduction}, ...]
    shortfall_window_days: Optional[Tuple[int, int]] = None  # P10-P90 of first deficit day, e.g. (18, 27)
    shortfall_window_dates: Optional[Tuple[str, str]] = None  # Same window as YYYY-MM-DD dates

@dataclass
class ActionRecommendation:
    invoice_id: str
    customer_id: str
    action_type: str           # 'DIRECT_CHASE', 'EARLY_DISCOUNT_2%', 'SPLIT_PAYMENT', 'LEGAL_MSMED_NOTICE'
    description: str
    expected_shortfall_avoided: float  # In ₹
    cost_of_action: float              # In ₹
    net_benefit: float                 # Avoided - Cost
    confidence_score: float
