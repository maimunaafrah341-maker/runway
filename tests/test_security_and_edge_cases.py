"""
Security, Vulnerability & Edge-Case Test Suite for Runway.

Tests:
1. CSV / Spreadsheet Formula Injection (=cmd|', @SUM, etc.)
2. Chronological Inversions (due_date < issue_date, payment < issue_date)
3. Financial Ledger Anomalies (negative invoices, duplicate IDs, extreme outliers)
4. PII Masking (GSTIN & Bank Account masking)
5. Engine Edge Cases (0 open invoices, empty ledger, negative starting cash)
"""
import unittest
import pandas as pd
import numpy as np

from data.sanitizer import (
    sanitize_formula_injection,
    sanitize_invoice_ledger,
    mask_gstin,
    mask_bank_account,
)
from engine.survival_model import InvoiceSurvivalModel
from engine.monte_carlo import CashFlowSimulator
from data.schemas import SEGMENT_TIER1_OEM, SEGMENT_MID_TIER, SEGMENT_RETAIL


class TestSecurityAndInputSanitization(unittest.TestCase):

    def test_formula_injection_defense(self):
        """Checks that dangerous formula injection payloads are neutralized with apostrophes."""
        malicious_inputs = [
            "=cmd|' /C calc'!A0",
            "=1+1",
            "+12345",
            "-5+cmd",
            "@SUM(A1:A10)",
            "\tmalicious_tab",
        ]
        for payload in malicious_inputs:
            sanitized = sanitize_formula_injection(payload)
            self.assertTrue(
                sanitized.startswith("'"),
                f"Payload '{payload}' was not neutralized: got '{sanitized}'"
            )

        # Harmless strings should remain untouched
        safe_str = "Tata Motors Ltd"
        self.assertEqual(sanitize_formula_injection(safe_str), safe_str)

    def test_pii_masking(self):
        """Checks that GSTIN and bank account numbers are properly masked."""
        real_gstin = "36AAAAA0000A1Z5"
        masked_gst = mask_gstin(real_gstin)
        self.assertEqual(masked_gst, "36*****1Z5")
        self.assertNotIn("0000", masked_gst)

        real_acc = "987654321098"
        masked_acc = mask_bank_account(real_acc)
        self.assertEqual(masked_acc, "********1098")
        self.assertNotIn("9876", masked_acc)

    def test_chronological_inversion_handling(self):
        """Checks that inverted dates (due_date < issue_date) are corrected automatically."""
        bad_df = pd.DataFrame([{
            "invoice_id": "INV-MALFORMED-1",
            "customer_id": "CUST-001",
            "customer_segment": "Tier-1 Auto OEM",
            "amount": 250000.0,
            "issue_date": "2025-06-15",
            "due_date": "2025-05-01",  # Inverted: before issue date!
            "actual_payment_date": "2025-05-10",  # Inverted: before issue date!
            "payment_terms_days": 30,
            "is_disputed": False,
        }])
        clean_df, warnings = sanitize_invoice_ledger(bad_df)
        self.assertGreater(len(warnings), 0)
        self.assertGreaterEqual(clean_df.iloc[0]["due_date"], clean_df.iloc[0]["issue_date"])
        self.assertGreaterEqual(clean_df.iloc[0]["actual_payment_date"], clean_df.iloc[0]["issue_date"])

    def test_negative_amount_and_deduplication(self):
        """Checks that duplicate IDs are deduplicated and negative values are sanitized."""
        dirty_df = pd.DataFrame([
            {
                "invoice_id": "INV-DUP",
                "customer_id": "CUST-A",
                "customer_segment": "Retail Distributor",
                "amount": -150000.0,  # Negative amount (credit note)
                "issue_date": "2025-01-01",
                "due_date": "2025-01-31",
            },
            {
                "invoice_id": "INV-DUP",  # Duplicate ID
                "customer_id": "CUST-A",
                "customer_segment": "Retail Distributor",
                "amount": -150000.0,
                "issue_date": "2025-01-01",
                "due_date": "2025-01-31",
            }
        ])
        clean_df, warnings = sanitize_invoice_ledger(dirty_df)
        self.assertEqual(len(clean_df), 1, "Duplicate invoice was not removed")
        self.assertEqual(clean_df.iloc[0]["amount"], 150000.0, "Negative amount was not corrected to positive")


class TestEngineEdgeCases(unittest.TestCase):

    def test_zero_open_invoices(self):
        """The Monte Carlo simulator should not crash if there are zero open receivables."""
        empty_invoices = pd.DataFrame(columns=[
            "invoice_id", "customer_id", "customer_segment", "amount",
            "issue_date", "due_date", "payment_terms_days", "is_disputed"
        ])
        empty_probs = np.zeros((0, 60))
        expenses_df = pd.DataFrame([{
            "expense_id": "EXP-1", "category": "Payroll", "amount": 100000.0,
            "due_date": "2026-01-07", "is_flexible": False
        }])

        simulator = CashFlowSimulator(n_simulations=100, horizon_days=60, rho=0.5, seed=42)
        result = simulator.run(
            initial_cash=500000.0,
            open_invoices_df=empty_invoices,
            payment_probs=empty_probs,
            expenses_df=expenses_df,
            as_of_date="2026-01-01",
            compute_risk_drivers=False,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.top_risk_invoices), 0)

    def test_negative_starting_balance(self):
        """Simulation should correctly report 100% deficit risk if starting balance is already negative."""
        empty_invoices = pd.DataFrame(columns=["invoice_id", "customer_id", "amount", "due_date"])
        empty_probs = np.zeros((0, 60))
        expenses_df = pd.DataFrame()

        simulator = CashFlowSimulator(n_simulations=100, horizon_days=60, rho=0.5, seed=42)
        result = simulator.run(
            initial_cash=-50000.0,  # Starting in the red
            open_invoices_df=empty_invoices,
            payment_probs=empty_probs,
            expenses_df=expenses_df,
            as_of_date="2026-01-01",
            compute_risk_drivers=False,
        )
        self.assertEqual(result.p_deficit_14d, 1.0)
        self.assertEqual(result.p_deficit_30d, 1.0)


if __name__ == "__main__":
    unittest.main()
