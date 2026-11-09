# Runway (Predict. Explain. Act.) — Engineering Playbook & Master Roadmap

> **Innovasia 2K26 — PS-04: Transaction-Aware Cash-Shortfall Early Warning for MSME Manufacturers**  
> **Team Fin-Hack**

---

## 1. Architectural Blueprint & Repository Layout

To prevent Claude Code or any AI assistant from drifting, hallucinating inconsistent schemas, or writing spaghetti monolithic code, adhere strictly to this modular design. Every module has a **single responsibility** and a **strict interface contract**.

```
runway/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── generator.py          # Synthetic MSME ledger generator (GST cycles, 45-day MSMED rules)
│   ├── schemas.py            # Pydantic / TypedDict data schemas
│   └── sample_ledger.csv     # Pre-generated realistic test dataset
├── engine/
│   ├── __init__.py
│   ├── survival_model.py     # Lifelines Cox Proportional Hazards + Weibull AFT fallback
│   ├── monte_carlo.py        # Vectorized 5000+ path cash simulation & correlated lateness
│   ├── counterfactual.py     # Top-5 risk driver attribution (what if invoice X paid on time)
│   └── optimizer.py          # Intervention ranking: Max (₹ Shortfall Avoided - Cost)
├── evaluation/
│   ├── __init__.py
│   ├── baselines.py          # 3 Baselines: Naive Due-Date, Moving Average, ARIMA(1,1,0)
│   └── backtest.py           # Rolling backtest, Brier score calibration & reliability curve
├── app/
│   ├── __init__.py
│   ├── main.py               # Streamlit application entry point
│   ├── components/
│   │   ├── fan_chart.py      # Plotly P10/P50/P90 cash-flow fan chart with deficit threshold
│   │   ├── risk_drivers.py   # Top-5 invoice risk breakdown & counterfactual delta chart
│   │   ├── action_cards.py   # Intervention cards (Chase, 2% Cash Discount, Renegotiate)
│   │   ├── what_if.py        # Real-time what-if scenario sliders
│   │   └── alert_modal.py    # Vernacular WhatsApp/SMS templates (English, Hindi, Telugu)
│   └── styles.css            # Clean, modern styling
└── tests/
    ├── test_generator.py
    ├── test_survival.py
    └── test_simulation.py
```

---

## 2. Core Data Contracts (`data/schemas.py`)

Both teammates agree on these data structures before writing a single line of business logic:

```python
# data/schemas.py
from dataclasses import dataclass
from typing import List, Optional
import datetime

@dataclass
class InvoiceRecord:
    invoice_id: str
    customer_id: str
    customer_segment: str     # 'Tier-1 Enterprise', 'Mid-tier OEM', 'Retailer'
    amount: float             # In INR (₹)
    issue_date: str           # YYYY-MM-DD
    due_date: str             # YYYY-MM-DD
    actual_payment_date: Optional[str] # Null if unpaid / right-censored
    payment_terms_days: int   # Usually 30, 45 (MSMED Act limit), or 60
    is_disputed: bool
    customer_past_avg_delay: float

@dataclass
class ExpenseObligation:
    expense_id: str
    category: str             # 'Payroll', 'GST/Tax', 'Raw Material', 'EMI/Rent'
    amount: float
    due_date: str             # YYYY-MM-DD
    is_flexible: bool         # Can payment be delayed by 7 days without shutdown?

@dataclass
class SimulationResult:
    trajectories: list        # 5000 paths x 60 days matrix
    dates: List[str]
    p_deficit_14d: float      # Probability of cash <= 0 within 14 days
    p_deficit_30d: float      # Probability within 30 days
    p_deficit_60d: float      # Probability within 60 days
    expected_shortfall_date: Optional[str]
    p10_trajectory: List[float]
    p50_trajectory: List[float]
    p90_trajectory: List[float]
    top_risk_invoices: List[dict] # {invoice_id, customer, amount, marginal_risk_reduction}

@dataclass
class ActionRecommendation:
    invoice_id: str
    customer_id: str
    action_type: str          # 'EARLY_DISCOUNT_2%', 'DIRECT_FOLLOWUP_CHASE', 'SPLIT_PAYMENT'
    expected_shortfall_avoided: float # In ₹
    cost_of_action: float             # In ₹ (e.g. 2% discount amount or relationship cost)
    net_benefit: float                # Avoided - Cost
    confidence_score: float
```

---

## 3. Work Division: 2-Person Parallel Execution

| Dimension | **Person 1 (Maimuna)**<br> *Quant & Machine Learning Engine* | **Person 2 (Teammate)**<br> *Product, UI & Decision Interface* |
| :--- | :--- | :--- |
| **Domain** | Data Generation, Survival Math, Monte Carlo Engine, Optimization | Streamlit App, Plotly Fan Charts, What-If Slider, Vernacular Alerts |
| **Day 1: Foundation** | 1. Implement `generator.py` (synthetic Indian MSME data).<br>2. Build `survival_model.py` (Lifelines Cox PH + Weibull). | 1. Setup Streamlit skeleton & design layout.<br>2. Build synthetic mock reader & state management. |
| **Day 2: Core Loop** | 1. Implement `monte_carlo.py` (5000 paths, correlated customer delays).<br>2. Build counterfactual risk attribution. | 1. Build Plotly Cash Fan Chart (P10/P50/P90).<br>2. Build Top-5 Risk Driver cards with visual bar chart. |
| **Day 3: Impact & Polish**| 1. Build `optimizer.py` (Net ₹ benefit ranking).<br>2. Run `backtest.py` (Brier score vs ARIMA & Naive). | 1. Implement Action cards with "Simulate Intervention" toggle.<br>2. Vernacular Hindi/Telugu/English WhatsApp modal. |

---

## 4. "No-Drift" Claude Code Prompt Playbook

When using Claude Code, **never** give broad prompts like *"Build the simulation and UI"*. Instead, copy-paste these exact, bounded task prompts one by one.

### --- PERSON 1 PROMPTS (Backend & Math Engine) ---

#### Prompt 1.1: Synthetic MSME Ledger Generator
```markdown
Context: We are building "Runway", a cash-shortfall early warning engine for Indian MSME manufacturers.
Task: Create `data/generator.py` and `data/schemas.py`.

Requirements:
1. Generate realistic synthetic accounting ledgers for an Indian MSME (auto-component or textile manufacturer with ₹5-15 Cr annual turnover).
2. Fields for invoices: invoice_id, customer_id, segment ('Tier-1 Auto OEM', 'Mid-tier Supplier', 'Retail Distributor'), amount (₹50,000 to ₹15,00,000), issue_date, due_date (terms 30, 45, 60 days according to MSMED Act 2006), payment_date (can be delayed, on time, or null if unpaid/censored), customer_past_avg_delay.
3. Realistic Indian MSME behaviors:
   - Tier-1 OEMs delay by 15-45 days past due date (high payment security, slow bureaucracy).
   - GST cycle outflow on the 20th of every month (GSTR-3B).
   - Monthly payroll on the 1st and 7th.
   - Correlated customer lateness: if a customer is cash-strapped, all their active invoices slip by a correlated factor gamma ~ LogNormal(mu, sigma).
4. Unpaid invoices up to current evaluation date should have `actual_payment_date = None` (right-censored observations).
5. Output function: `generate_msme_ledger(n_customers=25, n_invoices=500, start_date='2025-01-01', seed=42) -> tuple[pd.DataFrame, pd.DataFrame, float]`, returning (invoices_df, recurring_expenses_df, starting_cash_balance).
Include a `if __name__ == '__main__':` block that saves `sample_ledger.csv` and prints summary statistics.
```

#### Prompt 1.2: Survival Delay Modeling (`engine/survival_model.py`)
```markdown
Context: In Runway, invoice payment timeline is framed as survival analysis (time-to-event: payment received). Unpaid invoices are right-censored.
Task: Create `engine/survival_model.py`.

Requirements:
1. Use `lifelines.CoxPHFitter` as primary model to predict the survival curve S(t) = P(Payment arrives after t days from due date).
2. Features to use: `log_amount`, `customer_segment_encoded`, `payment_terms_days`, `past_avg_delay`, `is_month_end`.
3. Implement fallback: If Cox PH assumptions fail (e.g. non-convergence or test_proportional_hazards fails), cleanly catch and fit `lifelines.WeibullAFTFitter`.
4. Handle cold-start: For new customers with zero past history, use an empirical prior conditioned on `customer_segment`.
5. Class structure:
   class InvoiceSurvivalModel:
       def fit(self, historical_df: pd.DataFrame) -> None: ...
       def predict_cumulative_payment_prob(self, active_invoices_df: pd.DataFrame, max_days: int = 60) -> np.ndarray:
           """Returns matrix shape (n_invoices, max_days) where cell [i, d] is P(invoice i is paid by day d)."""
6. Add unit test assertions in `if __name__ == '__main__':`.
```

#### Prompt 1.3: Monte Carlo Cash Flow Simulation (`engine/monte_carlo.py`)
```markdown
Context: We need to propagate invoice payment uncertainty into 5,000 future cash-flow paths across 60 days.
Task: Create `engine/monte_carlo.py`.

Requirements:
1. Accept:
   - `initial_cash`: starting balance (e.g. ₹8,50,000)
   - `active_invoices`: open receivables with survival probabilities from `survival_model.py`
   - `recurring_expenses`: fixed outflows (payroll, GST, rent, vendor dues)
   - `n_simulations`: default 5000
   - `horizon_days`: 60
2. Correlated Lateness Engine:
   - In each simulation path `s`, sample a customer-specific delay shock: z_c ~ Normal(0, sigma_customer).
   - If z_c > 0, customer c's payment CDF shifts later for all their active invoices in that path.
3. Path calculation (vectorized with NumPy):
   - Daily Cash(t) = Cash(t-1) + Inflows(t) - Outflows(t).
4. Metrics to compute:
   - Deficit probability at 14, 30, and 60 days: P(Cash < 0).
   - P10, P50, P90 cash balance envelopes over 60 days.
   - Earliest and most likely deficit date interval (e.g. "Day 18 to 27").
5. Counterfactual Invoice Risk Ranking:
   - Run a marginal sensitivity: for each open invoice, rerun Monte Carlo setting this invoice as paid strictly on due date.
   - Marginal Risk Reduction = Baseline P(Deficit 30d) - Counterfactual P(Deficit 30d).
   - Return Top-5 Risk Driver invoices.
```

#### Prompt 1.4: Intervention Optimization & Baselines (`engine/optimizer.py` & `evaluation/`)
```markdown
Context: Runway must recommend ranked interventions and prove itself against 3 required baselines.
Task: Create `engine/optimizer.py` and `evaluation/backtest.py`.

Requirements:
1. `optimizer.py`:
   - Evaluate 3 possible interventions per high-risk invoice:
     a. `CHASE_NOW`: Direct phone/WhatsApp nudge. Response boost: +30% earlier payment; Cost: ₹500 (communication/time overhead).
     b. `DYNAMIC_DISCOUNT_2%`: Offer 2% discount if paid within 7 days. Response boost: +70% payment within 7 days; Cost: 2% of invoice amount.
     c. `SPLIT_PAYMENT`: Offer 50% now, 50% in 30 days.
   - Rank interventions by: `Expected Shortfall Avoided (₹) - Cost of Intervention (₹)`.
2. `evaluation/backtest.py`:
   - Implement 3 Baselines:
     a. Baseline 1: Naive Due-Date (assumes 100% payment on invoice due date).
     b. Baseline 2: Moving Average Historical Delay (shifts due date by customer's mean delay).
     c. Baseline 3: ARIMA(1,1,0) time series model on daily aggregate net cashflow.
   - Run rolling backtest over past 90 days.
   - Compute Brier Score: (Predicted Probability - Actual Deficit Indicator)^2.
   - Output comparison table showing Runway beats baselines in Warning Lead Time (days) and Brier Score.
```

---

### --- PERSON 2 PROMPTS (Streamlit UI & Product Experience) ---

#### Prompt 2.1: Streamlit Layout & Visual Identity (`app/main.py`)
```markdown
Context: Build the presentation dashboard for Runway: Transaction-Aware Cash-Shortfall Early Warning for MSME Manufacturers.
Task: Create `app/main.py` using Streamlit and Plotly.

Requirements:
1. Visual Header:
   - Title: "Runway (Predict. Explain. Act.)"
   - Subtitle: "Cash-Shortfall Early Warning System for MSME Manufacturers | PS-04"
   - Top Metric Bar:
     - Metric 1: Current Cash In Hand (₹)
     - Metric 2: 14-Day Shortfall Risk (%) [Color green if <20%, amber if 20-50%, red if >50%]
     - Metric 3: 30-Day Shortfall Risk (%)
     - Metric 4: Projected Deficit Window (e.g. "Day 19 – Day 26")
2. Three Core Sections corresponding to the pitch:
   - Tab 1: "1. PREDICT (Cash Flow Horizon)"
   - Tab 2: "2. EXPLAIN (Top Risk Drivers)"
   - Tab 3: "3. ACT (Prioritized Interventions)"
   - Tab 4: "EVALUATION (Baselines & Calibration)"
3. Sidebar controls:
   - Select Scenario: "Normal Operations", "Major OEM Delay (Tata Auto)", "End of Quarter Crunch".
   - Slider: "Starting Cash Buffer (₹ Lakhs)".
   - Button: "Run 5,000-Path Monte Carlo Simulation".
```

#### Prompt 2.2: Interactive Fan Chart & Risk Waterfall (`app/components/`)
```markdown
Context: Implement rich visualizations for Runway's Predict and Explain tabs.
Task: Create `app/components/fan_chart.py` and `app/components/risk_drivers.py`.

Requirements:
1. `fan_chart.py`:
   - Plotly chart displaying:
     - 60-day horizon on X-axis with actual dates.
     - P50 (median) cash trajectory line in bold navy blue.
     - P10 to P90 shaded confidence cone (soft blue gradient).
     - Horizontal dashed red line at Cash = 0 (Deficit Danger Line).
     - Visual red highlight zone marking the date range where P10 drops below ₹0.
     - Annotations for major known outflow spikes (e.g., "GST Outflow ₹4.2L on 20th", "Payroll ₹6.5L on 1st").
2. `risk_drivers.py`:
   - Horizontal bar chart showing the Top-5 Invoices contributing to the shortfall.
   - Each bar displays: Customer Name, Invoice #, Invoice Amount (₹), and Risk Contribution (% marginal drop in shortfall probability).
   - Clickable or expandable details showing historical payment track record and days overdue.
```

#### Prompt 2.3: Actionable Decision Cards & Vernacular Alerts (`app/components/`)
```markdown
Context: MSME owners need clear actions, not raw numbers. They also need WhatsApp alerts in vernacular languages (Hindi/Telugu/English).
Task: Create `app/components/action_cards.py` and `app/components/alert_modal.py`.

Requirements:
1. `action_cards.py`:
   - Display top recommendations ranked by Net Benefit (₹ Shortfall Avoided - Cost).
   - Card layout:
     - Header: Action type (e.g., "Chase Tata Auto #482 immediately", "Offer 2% Cash Discount to Sharma Steel #319").
     - Impact Badge: "Avoids ₹2.4L shortfall risk | Cost: ₹4,800".
     - Button: "Simulate This Action" -> instantly recalculates and visually lifts the fan chart above ₹0.
2. `alert_modal.py`:
   - Generate ready-to-send WhatsApp / SMS message previews with language toggle:
     - English: "ALERT: Cash balance projected to dip below zero between 18th-24th Nov due to ₹3.2L delayed payment from Sharma Steel. Recommended action: Call Sharma Steel account head today."
     - Hindi: "चेतावनी: शर्मा स्टील से ₹3.2 लाख के भुगतान में देरी के कारण 18-24 नवंबर के बीच कैश की कमी होने की 64% संभावना है..."
     - Telugu: "హెచ్చరిక: శర్మ స్టీల్ నుండి ₹3.2 లక్షల చెల్లింపు ఆలస్యం కావడం వల్ల నవంబర్ 18-24 మధ్య నగదు కొరత ఏర్పడే అవకాశం ఉంది..."
   - Include a "Copy to Clipboard" button.
```

---

## 5. Judge-Winning Checklist for Innovasia 2K26

When demonstrating to judges, highlight these 4 specific technical achievements from your slide deck:

1. **Survival Analysis over Simple Classification:**  
   Most student teams just predict a binary *"Will invoice be late: Yes/No"*. You predict **when** it will arrive as a survival function $S(t)$, correctly handling right-censored unpaid invoices.
2. **Correlated Lateness (Customer Frailty):**  
   If an OEM delays payment, it delays **all** their invoices together. Your Monte Carlo explicitly models this joint correlation.
3. **Counterfactual Risk Tracing:**  
   You isolate the exact invoices causing the cash deficit using simulated counterfactuals ($P(\text{deficit}) - P(\text{deficit} \mid \text{invoice } k \text{ paid})$).
4. **Economic Action Ranking:**  
   Actions are ranked strictly by $\text{Net Benefit} = \mathbb{E}[\text{Deficit Avoided}] - \text{Cost of Action}$, directly answering *"What should the factory owner do first?"*
