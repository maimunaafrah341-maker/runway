# Runway (Team Fin-Hack) — 5-Minute Pitch Script & Q&A Defense Guide
**INNOVASIA 2K26 | Problem Statement: PS-04 (Fintech)**

---

## 1. Golden Rules for the 5-Minute Presentation

1. **The Ratio:** **1 Minute Slides + 3.5 Minutes Live Demo + 30 Seconds Close**.
2. **Browser Setup:**
   - Open your browser to `http://localhost:8501`.
   - Press `Ctrl` + `+` to zoom to **110% or 125%** so all text and metrics are large and crisp over screen sharing.
   - Close all unrelated tabs and notifications.
3. **Tone & Pace:** Speak with calm authority. Don't rush. Point your cursor at the numbers and chart pins as you mention them.

---

## 2. Timed 5-Minute Pitch Script (Second-by-Second)

### [0:00 – 1:00] The Human Story & Problem (On Slides)
> "Respected judges, in India today, over **₹7.34 Lakh Crore** is locked up in delayed payments to MSMEs.
> 
> Meet **Raghav**, proprietor of Sri Lakshmi Auto Components, a ₹12.6 Cr manufacturing factory in Hyderabad. Today, his bank balance shows **₹9.22 Lakhs**. He feels safe. 
> 
> But in exactly 6 days, on **January 7th**, Raghav’s factory wage checks will bounce because payments from his largest buyer are delayed in bureaucracy. By the time his account hits zero, it’s too late.
> 
> Conventional accounting software like Tally only flags overdue invoices in hindsight. By then, the factory faces a liquidity crisis. 
> 
> We built **Runway: Predict. Explain. Act.** to forecast cash shortages before they happen."

---

### [1:00 – 2:15] Switch to Live Screen: PREDICT (The Quant Breakthrough)
*(Switch screen share to your browser running Runway)*

> "Here is Runway's live cockpit for Raghav’s factory:
> 
> * **The Early Warning:** Look at the top metric cards. Runway flags an **89.0% shortfall risk** within 30 days and pinpoints the exact deficit window: **January 07 to January 10**.
> * **The 60-Day Cash Horizon Fan Chart:** Pointing to the chart, you can see our median cash trajectory (the neon cyan line) plunging to **-₹2.64 Lakhs on January 08**. 
> * Why does it crash? Runway overlays the exact obligations: **January 07 Factory Wages (-₹4.5L)** and **January 10 Raw Material vendor payments (-₹6.8L)**.
> 
> **Our Quantitative Edge:** Most student hackathon projects do binary 0/1 classification ('is this invoice late: yes or no'). But in manufacturing, timing is everything. 
> We model payments using **Lifelines Survival Analysis (Cox Proportional Hazards with Weibull AFT fallback)** to predict the arrival curve $S(t)$, and propagate joint uncertainty across **5,000 Monte Carlo paths**. 
> 
> Crucially, we use a **Gaussian Copula ($\rho = 0.5$)** to capture **correlated customer lateness**—when an OEM stalls, all their invoices slip together."

---

### [2:15 – 3:30] EXPLAIN & ACT (The Economic ROI)
*(Click on Tab 02: EXPLAIN)*

> "Raghav doesn’t need a black-box score; he needs to know who is causing the crisis.
> 
> Our counterfactual simulation isolates that **Brahmaputra Mobility Ltd** accounts for 3 of the top 5 delayed receivables—representing **₹17.50 Lakhs** of trapped capital. When their bureaucratic payment cycle freezes, it pulls down his entire payroll.
> 
> *(Click on Tab 03: ACT)*
> 
> Runway doesn’t give vague advice. It ranks interventions strictly by **Net Economic Benefit ($\mathbb{E}[\text{Deficit Avoided}] - \text{Cost}$)**:
> 
> * **Action #1:** Offer a 2% early cash settlement discount to Brahmaputra Mobility on invoice INV-2025-0357. This avoids **₹3.79 Lakhs** of deficit risk at a cost of just **₹15,900**—delivering a **net liquidity gain of ₹3.63 Lakhs (a 23x ROI)**!
> * **Action #2:** For invoices overdue past 45 days, Runway leverages **Sections 15–16 of the Indian MSMED Act 2006** to generate a statutory notice claiming **3x RBI compound interest**, recovering dead capital where routine phone chasing fails."

---

### [3:30 – 4:30] Localization & Empirical Benchmarks
*(Point to the WhatsApp alert box on Tab 03)*

> "Factory proprietors don't read raw statistical charts. With one click, Raghav can dispatch this localized advisory directly to his phone or buyer over WhatsApp in **Telugu or Hindi**.
> 
> *(Click on Tab 04: BENCHMARK)*
> 
> Under the hood, rolling 90-day backtests prove Runway achieves a **0.234 Brier score**—outperforming ARIMA(1,1,0) at 0.283 and naive due-date rules at 0.389—while providing **22 days of actionable warning lead time** with **30% fewer false alarms** than time-series baselines."

---

### [4:30 – 5:00] Conclusion
> "Runway connects uncertain invoice timelines directly to factory survival: **Predict. Explain. Act.** 
> Thank you, and we look forward to your questions."

---

## 3. The 5-Minute Q&A Defense Cheat Sheet (Master These 6 Answers)

Keep every answer **under 45 seconds**. Leave your screen shared on the dashboard and point to the charts!

#### Q1: "Why survival analysis instead of standard machine learning (XGBoost / Logistic Regression)?"
> **Answer:** *"Logistic regression only predicts a binary Yes/No: 'will it be late'. But in cash flow, timing is everything—paying 3 days late is harmless; paying 25 days late causes payroll failure. Furthermore, open invoices at the evaluation date are right-censored (we only know they haven't paid yet). Survival analysis models the full continuous time-to-arrival curve $S(t)$, correctly handling censored data without lookahead leakage."*

#### Q2: "How do you handle new customers with zero payment history (cold start)?"
> **Answer:** *"In `engine/survival_model.py`, when a new buyer has zero transaction history, we impute their delay prior using the empirical segment-level mean (Tier-1 OEM vs Mid-Tier vs Retail). As they complete orders, their individual empirical delay dynamically updates and refines the hazard rate."*

#### Q3: "Why did you use a Gaussian Copula for Monte Carlo instead of independent draws?"
> **Answer:** *"If you assume invoice payments are independent, the Law of Large Numbers unrealistically smooths out risk across 100 invoices. In reality, MSME payments are correlated: if an OEM freezes disbursements, all their open invoices slip together. Using a Gaussian Copula with correlation $\rho = 0.5$ captures this joint frailty, revealing that a single buyer accounts for 3 of our top 5 risk drivers."*

#### Q4: "How does your intervention ranking model work?"
> **Answer:** *"We rank actions strictly by $\text{Net Benefit} = \mathbb{E}[\text{Deficit Avoided}] - \text{Cost of Action}$. For example, offering a 2% early cash discount costs ₹15,900 but avoids ₹3.79 Lakhs in expected shortfall, delivering a net gain of ₹3.63 Lakhs (23x ROI). We only recommend an action if its expected avoided shortfall exceeds its implementation cost."*

#### Q5: "How does the Indian regulatory context (MSMED Act 2006) factor in?"
> **Answer:** *"Sections 15–16 of the MSMED Act require buyers to pay MSMEs within 45 days. Beyond 45 days, compound interest at 3x the RBI bank rate is mandatory. In Tab 03, Runway flags statutory breaches and drafts legal demand notices that recover aged receivables where routine reminders fail."*

#### Q6: "How do you prove the model is well-calibrated?"
> **Answer:** *"We ran a rolling backtest over 90 days evaluating 30-day forward windows. Runway achieved a 0.234 Brier score (beating Naive at 0.389 and ARIMA at 0.283) and an Expected Calibration Error of 0.231. Furthermore, 5-fold cross-validation tuned the Cox L2 penalizer to $\alpha = 0.1$, achieving a concordance index of 0.704."*

---

## 4. Offline Round Checklist (Day After Tomorrow)

1. **Test Offline:** Disconnect your laptop from Wi-Fi and verify the command runs locally:
   `.\venv\Scripts\python.exe -m streamlit run app/main.py`
2. **Hardware:** Bring your laptop charger and an **HDMI to USB-C adapter**.
3. **Physical Handout:** Print 3 physical copies of your 1-page overview (Slides 1 & 2) to hand to the judges as you walk up. Judges love having something tangible to hold!
