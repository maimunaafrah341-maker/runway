"""
Runway — Executive CLI Live Demonstration.

Run this script anytime to see the full end-to-end engine in action:
    python demo.py
"""
import sys
from pathlib import Path

# Fix Windows console encoding for Rupee symbol and Indian vernacular text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine import run_runway_pipeline


def format_inr(val: float) -> str:
    """Formats values in Indian Lakhs (L) or Crores (Cr)."""
    abs_v = abs(val)
    sign = "-" if val < 0 else ""
    if abs_v >= 10_000_000:
        return f"{sign}₹{abs_v / 10_000_000:.2f} Cr"
    elif abs_v >= 100_000:
        return f"{sign}₹{abs_v / 100_000:.2f} L"
    return f"{sign}₹{abs_v:,.0f}"


def run_cli_demo():
    print("=" * 72)
    print("   RUNWAY: TRANSACTION-AWARE CASH-SHORTFALL EARLY WARNING (PS-04)   ")
    print("   Team Fin-Hack | Innovasia 2K26                                   ")
    print("=" * 72)
    print("\n[+] Initializing Runway Quant Engine & Simulating 5,000 Paths...")

    # Execute end-to-end pipeline
    sim_result, actions, simulator = run_runway_pipeline(
        as_of_date="2026-01-01",
        n_simulations=5000,
        horizon_days=60,
        rho=0.5,
        seed=42,
    )

    print("\n" + "—" * 72)
    print("1. PREDICT — PROBABILISTIC DEFICIT FORECAST")
    print("—" * 72)
    print(f"  • Starting Cash Balance:    {format_inr(922000.0)}")
    print(f"  • 14-Day Shortfall Risk:    {sim_result.p_deficit_14d:.1%}")
    print(f"  • 30-Day Shortfall Risk:    {sim_result.p_deficit_30d:.1%}  <-- CRITICAL WARNING")
    print(f"  • 60-Day Shortfall Risk:    {sim_result.p_deficit_60d:.1%}")
    if sim_result.shortfall_window_days:
        w_days = sim_result.shortfall_window_days
        w_dates = sim_result.shortfall_window_dates
        print(f"  • Projected Deficit Window: Day {w_days[0]} to Day {w_days[1]} ({w_dates[0]} to {w_dates[1]})")
        print("    [Trigger: Wages on 7th + Raw Material vendor payments on 10th]")

    print(f"\n  Cash Balance Forecast Bands:")
    print(f"    - Day 30: P10 = {format_inr(sim_result.p10_trajectory[29])} | P50 = {format_inr(sim_result.p50_trajectory[29])} | P90 = {format_inr(sim_result.p90_trajectory[29])}")
    print(f"    - Day 60: P10 = {format_inr(sim_result.p10_trajectory[59])} | P50 = {format_inr(sim_result.p50_trajectory[59])} | P90 = {format_inr(sim_result.p90_trajectory[59])}")

    print("\n" + "—" * 72)
    print("2. EXPLAIN — TOP RISK-DRIVING INVOICES (COUNTERFACTUAL SIMULATION)")
    print("—" * 72)
    print(f"  {'Invoice ID':<15} {'Customer Name':<30} {'Amount':<12} {'Risk Drop'}")
    print("  " + "-" * 68)
    for inv in sim_result.top_risk_invoices:
        print(f"  {inv['invoice_id']:<15} {inv['customer'][:28]:<30} {format_inr(inv['amount']):<12} -{inv['marginal_risk_reduction']:.1%}")
    print("  [Insight: 3 of the top 5 belong to Brahmaputra Mobility due to correlated OEM delays]")

    print("\n" + "—" * 72)
    print("3. ACT — ECONOMICALLY RANKED INTERVENTIONS (MAX NET BENEFIT)")
    print("—" * 72)
    print(f"  {'Action Type':<22} {'Target Invoice':<15} {'Deficit Avoided':<18} {'Cost':<10} {'Net Benefit'}")
    print("  " + "-" * 75)
    for act in actions:
        print(
            f"  {act.action_type:<22} {act.invoice_id:<15} {format_inr(act.expected_shortfall_avoided):<18} "
            f"{format_inr(act.cost_of_action):<10} {format_inr(act.net_benefit)}"
        )

    print("\n" + "—" * 72)
    print("4. VERNACULAR DECISION ALERTS (FOR FACTORY PROPRIETOR)")
    print("—" * 72)
    top_act = actions[0] if actions else None
    print("  [ENGLISH]")
    print(f"  URGENT: Projected cash deficit between Jan 07-10. Top action: Offer 2% early discount")
    print(f"  on invoice {top_act.invoice_id if top_act else 'N/A'}. Net savings: {format_inr(top_act.net_benefit if top_act else 0)}.")

    print("\n  [HINDI / हिंदी]")
    print(f"  चेतावनी: 7-10 जनवरी के बीच कैश की कमी होने की {sim_result.p_deficit_30d:.0%} संभावना है।")
    print(f"  सुझाव: ब्रह्मपुत्र मोबिलिटी को बिल {top_act.invoice_id if top_act else ''} पर 2% शीघ्र छूट दें। शुद्ध बचत: {format_inr(top_act.net_benefit if top_act else 0)}।")

    print("\n  [TELUGU / తెలుగు]")
    print(f"  హెచ్చరిక: జనవరి 7-10 మధ్య నగదు కొరత ఏర్పడే ప్రమాదం {sim_result.p_deficit_30d:.0%} ఉంది.")
    print(f"  సిఫార్సు: ఇన్వాయిస్ {top_act.invoice_id if top_act else ''} పై 2% త్వరిత తగ్గింపు ఇవ్వండి. నికర ప్రయోజనం: {format_inr(top_act.net_benefit if top_act else 0)}.")

    print("\n" + "=" * 72)
    print("   [DEMO COMPLETE: ENGINE READY FOR STREAMLIT UI INTEGRATION]       ")
    print("=" * 72)


if __name__ == "__main__":
    run_cli_demo()
