# Runway: Transaction-Aware Cash-Shortfall Early Warning for MSME Manufacturers
**INNOVASIA 2K26 | Problem Statement: PS-04 | Team Fin-Hack**

Runway is a probabilistic cash-flow intelligence platform designed for Indian MSME manufacturers. Unlike conventional accounting tools that only flag overdue invoices in hindsight, Runway connects invoice payment uncertainty to future cash shortages across 14, 30, and 60-day horizons.

## Core Features (Predict. Explain. Act.)
1. **Predict:** Lifelines Cox Survival Model with Weibull AFT fallback + 5,000-path Monte Carlo simulation with correlated customer delays.
2. **Explain:** Counterfactual risk attribution to isolate top-5 shortfall-inducing invoices.
3. **Act:** Economic intervention ranking ($\mathbb{E}[\text{Deficit Avoided}] - \text{Cost}$) with plain-language alerts in English, Hindi, and Telugu.
4. **Evaluate:** Rolling backtests and Brier score calibration evaluated against 3 required baselines.

## Quick Start
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run synthetic data generator
python data/generator.py

# Launch Streamlit dashboard
streamlit run app/main.py
```
