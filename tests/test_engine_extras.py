"""
Tests for the joint "what if these invoices are paid on time" counterfactual.

Run: python -m unittest tests.test_engine_extras -v
"""
import unittest

import numpy as np
import pandas as pd

from data.generator import generate_msme_ledger
from engine import run_runway_pipeline


class TestJointCounterfactual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inv, exp, cash = generate_msme_ledger(seed=42)
        cls.sim, _, cls.simulator = run_runway_pipeline(
            inv, exp, cash, as_of_date=inv.attrs["as_of_date"], n_simulations=2000, horizon_days=60, seed=42)
        cls.ids = [d["invoice_id"] for d in cls.sim.top_risk_invoices]

    def test_single_invoice_matches_attribution(self):
        """One invoice: same number the risk-driver table reports."""
        top = self.sim.top_risk_invoices[0]
        joint = self.simulator.joint_counterfactual_risk([top["invoice_id"]])
        self.assertAlmostEqual(joint, self.sim.p_deficit_30d - top["marginal_risk_reduction"], places=9)

    def test_more_invoices_never_raise_risk(self):
        risks = [self.simulator.joint_counterfactual_risk(self.ids[:k]) for k in range(1, len(self.ids) + 1)]
        self.assertTrue(all(r <= self.sim.p_deficit_30d + 1e-12 for r in risks))
        self.assertLess(risks[-1], self.sim.p_deficit_30d)

    def test_empty_list_is_baseline(self):
        self.assertAlmostEqual(self.simulator.joint_counterfactual_risk([]), self.sim.p_deficit_30d, places=9)

    def test_unknown_invoice_is_an_error(self):
        with self.assertRaises(ValueError):
            self.simulator.joint_counterfactual_risk(["INV-DOES-NOT-EXIST"])


if __name__ == "__main__":
    unittest.main()
