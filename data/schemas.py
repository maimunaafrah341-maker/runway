"""
Data contracts and schema definitions for Runway.
Both backend modeling and frontend UI rely strictly on these structures.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InvoiceRecord:
    invoice_id: str
    customer_id: str
    customer_segment: str      # 'Tier-1 Enterprise', 'Mid-tier OEM', 'Retailer'
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
    category: str              # 'Payroll', 'GST/Tax (GSTR-3B)', 'Raw Material', 'EMI/Rent'
    amount: float
    due_date: str              # YYYY-MM-DD
    is_flexible: bool          # Can payment be delayed by 7 days without shutdown?

@dataclass
class SimulationResult:
    trajectories: list         # Matrix of 5000 paths x 60 days
    dates: List[str]
    p_deficit_14d: float       # Probability of cash <= 0 within 14 days
    p_deficit_30d: float       # Probability within 30 days
    p_deficit_60d: float       # Probability within 60 days
    expected_shortfall_date: Optional[str]
    p10_trajectory: List[float]
    p50_trajectory: List[float]
    p90_trajectory: List[float]
    top_risk_invoices: List[dict] # [{invoice_id, customer, amount, marginal_risk_reduction}, ...]

@dataclass
class ActionRecommendation:
    invoice_id: str
    customer_id: str
    action_type: str           # 'DIRECT_CHASE', 'CASH_DISCOUNT_2%', 'SPLIT_PAYMENT'
    description: str
    expected_shortfall_avoided: float  # In ₹
    cost_of_action: float              # In ₹
    net_benefit: float                 # Avoided - Cost
    confidence_score: float
