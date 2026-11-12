"""
Runway Engine Package.

Exports core modeling classes and a unified `run_runway_pipeline()` helper
that executes the end-to-end simulation and intervention optimization.
"""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from engine.survival_model import InvoiceSurvivalModel
from engine.monte_carlo import CashFlowSimulator
from engine.optimizer import InterventionOptimizer
from data.schemas import SimulationResult, ActionRecommendation

__all__ = [
    "InvoiceSurvivalModel",
    "CashFlowSimulator",
    "InterventionOptimizer",
    "run_runway_pipeline",
]

def run_runway_pipeline(
    invoices_df: Optional[pd.DataFrame] = None,
    expenses_df: Optional[pd.DataFrame] = None,
    starting_cash: Optional[float] = None,
    as_of_date: str = "2026-01-01",
    n_simulations: int = 5000,
    horizon_days: int = 60,
    rho: float = 0.5,
    seed: int = 42,
) -> Tuple[SimulationResult, List[ActionRecommendation], CashFlowSimulator]:
    """Runs the end-to-end Runway pipeline:
    1. Loads or accepts data
    2. Fits InvoiceSurvivalModel on closed/historical invoices
    3. Predicts cumulative payment probabilities for open invoices
    4. Simulates 5,000 Monte Carlo paths with customer correlation rho
    5. Runs counterfactual attribution and intervention optimization

    Returns:
        (simulation_result, recommended_actions, simulator_instance)
    """
    if invoices_df is None:
        # No ledger supplied: run on the synthetic demo company.
        from data.generator import generate_msme_ledger
        invoices_df, gen_expenses, gen_cash = generate_msme_ledger(seed=seed)
        expenses_df = gen_expenses if expenses_df is None else expenses_df
        starting_cash = gen_cash if starting_cash is None else starting_cash
    elif expenses_df is None or starting_cash is None:
        # Never silently swap a user's real ledger for synthetic data.
        raise ValueError("When invoices_df is supplied, expenses_df and starting_cash are required too")

    # Open invoices: those without actual_payment_date
    open_invoices = invoices_df[invoices_df["actual_payment_date"].isna()].copy()

    # 1. Fit survival model
    survival_model = InvoiceSurvivalModel(prefer="cox")
    survival_model.fit(invoices_df)

    # 2. Predict payment probability curves
    probs = survival_model.predict_cumulative_payment_prob(
        open_invoices, max_horizon_days=horizon_days, as_of_date=as_of_date
    )

    # 3. Simulate Monte Carlo paths
    simulator = CashFlowSimulator(
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        rho=rho,
        seed=seed,
    )
    sim_result = simulator.run(
        initial_cash=starting_cash,
        open_invoices_df=open_invoices,
        payment_probs=probs,
        expenses_df=expenses_df,
        as_of_date=as_of_date,
    )

    # 4. Optimize interventions
    optimizer = InterventionOptimizer()
    actions = optimizer.recommend_actions(
        simulator=simulator,
        open_invoices_df=open_invoices,
        top_risk_invoices=sim_result.top_risk_invoices,
        expenses_df=expenses_df,
        initial_cash=starting_cash,
        as_of_date=as_of_date,
        best_per_invoice=True,
    )

    return sim_result, actions, simulator
