"""
Tests for the Fund tab: loan maths on the simulated futures, invoice security, credit sources, and the
shared lending network (requests, answers, counter-offers, notifications state).

Run: python -m unittest tests.test_funding -v
"""
import threading
import unittest

import numpy as np
import pandas as pd

from data.generator import generate_msme_ledger
from engine import run_runway_pipeline
from funding import (ACCEPTED, COUNTERED, DECLINED, DEMO_LENDERS, WAITING, WITHDRAWN, LendingNetwork,
                     apply_loans, calendar_ics, check_loan, credit_dips, credit_sources, fit_checks, fits, interest,
                     lending_effect, loan_path, pick_security, safe_to_lend, split_amount, suggest_repay_day,
                     term_sheet, treds_invoices)


class TestLoanMaths(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inv, exp, cash = generate_msme_ledger(seed=42)
        cls.as_of = pd.Timestamp(inv.attrs["as_of_date"])
        sim, _, s = run_runway_pipeline(inv, exp, cash, as_of_date=cls.as_of, n_simulations=2000,
                                        horizon_days=60, seed=42)
        cls.T = np.asarray(sim.trajectories)
        cls.pay_days = s.payment_days_
        cls.open = inv.set_index("invoice_id").loc[s.open_invoice_ids_].reset_index()
        cls.need = float(-np.percentile(cls.T[:, :30].min(axis=1), 10))

    def test_interest_is_simple_actual_365(self):
        self.assertAlmostEqual(interest(3_650_000, 0.12, 10), 12_000.0)
        self.assertEqual(interest(1_000_000, 0.12, -5), 0.0)

    def test_loan_path_adds_then_repays(self):
        path = loan_path(self.T, 1_000_000, 4, 20, 0.12)
        diff = path - self.T
        np.testing.assert_allclose(diff[:, :4], 0)
        np.testing.assert_allclose(diff[:, 4:20], 1_000_000)
        np.testing.assert_allclose(diff[:, 20:], -interest(1_000_000, 0.12, 16))

    def test_loan_cuts_shortfall_risk(self):
        c = check_loan(self.T, self.need, 4, 30, 0.12)
        self.assertLess(c.p_short_with, c.p_short_without)
        self.assertTrue(0 <= c.p_repay_on_time <= 1)
        self.assertEqual(c.days, 26)
        with self.assertRaises(ValueError):
            check_loan(self.T, self.need, 10, 10, 0.12)

    def test_suggested_repay_day_is_best_in_range(self):
        start, rate = 4, 0.12
        best = suggest_repay_day(self.T, self.need, start, rate)
        self.assertTrue(start + 14 <= best <= start + 45)

        def score(d):
            p = loan_path(self.T, self.need, start, d, rate)[:, start:d + 8]
            return (p >= 0).all(axis=1).mean()
        self.assertGreaterEqual(score(best) + 1e-12, max(score(d) for d in range(start + 14, start + 46)))

    def test_security_is_disjoint_and_covers_margin(self):
        first = pick_security(self.open, self.pay_days, 30, 1_000_000)
        self.assertGreaterEqual(first.total, 1.25 * 1_000_000)
        self.assertTrue(0 <= first.cover <= 1)
        self.assertFalse(any(self.open.set_index("invoice_id").loc[[s["invoice_id"] for s in first.invoices],
                                                                   "is_disputed"]))
        second = pick_security(self.open, self.pay_days, 30, 1_000_000,
                               exclude=[s["invoice_id"] for s in first.invoices])
        self.assertFalse({s["invoice_id"] for s in first.invoices} & {s["invoice_id"] for s in second.invoices})

    def test_security_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            pick_security(self.open.iloc[:-1], self.pay_days, 30, 1_000_000)

    def test_treds_invoices_are_undisputed_not_due_big_buyers(self):
        tr = treds_invoices(self.open, self.as_of)
        self.assertGreater(len(tr), 0)
        self.assertTrue((tr["customer_segment"] == "Tier-1 Auto OEM").all())
        self.assertTrue((pd.to_datetime(tr["due_date"]) > self.as_of).all())
        self.assertFalse(tr["is_disputed"].any())

    def test_credit_sources(self):
        tr = treds_invoices(self.open, self.as_of)
        rows = credit_sources(3_000_000, 5, 27, self.as_of, tr, 3_000_000, [0.11, 0.12], 0.12)
        keys = [r.key for r in rows]
        self.assertEqual(keys, ["treds", "peer", "bank", "nbfc"])
        by = {r.key: r for r in rows}
        self.assertAlmostEqual(by["peer"].cost, interest(3_000_000, 0.12, 27))
        self.assertEqual(by["bank"].fits, "maybe")       # 3-7 days against 5 days left
        self.assertEqual(by["nbfc"].fits, "yes")
        self.assertGreater(by["nbfc"].cost, by["bank"].cost)
        no_peers = credit_sources(3_000_000, 5, 27, self.as_of, tr, 0, [])
        self.assertNotIn("peer", [r.key for r in no_peers])

    def test_safe_to_lend_keeps_the_cushion_in_9_of_10_futures(self):
        rich = self.T + 6_000_000                              # the same business with ₹60 L more cash
        safe = safe_to_lend(rich, 45, keep=500_000)
        self.assertGreater(safe, 0)
        after = rich[:, :45] - safe                            # lend `safe` for the whole period
        self.assertGreaterEqual((after.min(axis=1) >= 500_000 - 1).mean(), 0.9 - 1e-9)
        self.assertEqual(safe_to_lend(self.T, 45, keep=500_000), 0.0)   # Raghav has nothing to spare

    def test_lending_effect_raises_the_lenders_own_risk(self):
        rich = self.T + 6_000_000
        without, with_loan = lending_effect(rich, 2_000_000, 4, 31, 0.12)
        self.assertGreaterEqual(with_loan, without)
        self.assertEqual(lending_effect(rich, 0, 4, 31, 0.12), (without, without))

    def test_apply_loans_matches_single_loans_and_mirrors_lending(self):
        one = apply_loans(self.T, [(1_000_000, 4, 20, 0.12)])
        np.testing.assert_allclose(one, loan_path(self.T, 1_000_000, 4, 20, 0.12))
        two = apply_loans(self.T, [(1_000_000, 4, 20, 0.12), (500_000, 4, 30, 0.13)])
        np.testing.assert_allclose(two - self.T, (one - self.T) + (loan_path(self.T, 500_000, 4, 30, 0.13) - self.T))
        lent = apply_loans(self.T, [(-1_000_000, 4, 20, 0.12)])          # the lender's side of the same loan
        np.testing.assert_allclose((lent - self.T) + (one - self.T), 0, atol=1e-6)
        np.testing.assert_allclose(apply_loans(self.T, []), self.T)

    def test_credit_dips(self):
        dates = pd.date_range("2026-01-02", periods=8)
        dips = credit_dips([5, -1, -2, 3, 4, -1, 2, -3], dates)
        self.assertEqual([(a.day, b.day) for a, b in dips], [(3, 4), (7, 7), (9, 9)])


class TestLendingNetwork(unittest.TestCase):

    def setUp(self):
        self.net = LendingNetwork()
        self.kw = dict(borrower="Borrower Pvt Ltd", owner="Raghav", borrower_note="test",
                       start=pd.Timestamp("2026-01-06"), repay=pd.Timestamp("2026-02-02"),
                       interest=17_753.0, check={"p_repay_on_time": 0.74, "n_futures": 5000})

    def test_fit_checks_explain_each_condition(self):
        ganga = self.net.lender("ganga")
        self.assertTrue(fits(ganga, 27, 0.12, True, 0.74))
        self.assertFalse(fits(ganga, 27, 0.12, False, 0.74))     # wants security
        self.assertFalse(fits(ganga, 27, 0.10, True, 0.74))      # rate too low
        self.assertFalse(fits(ganga, 60, 0.12, True, 0.74))      # too long
        self.assertFalse(fits(ganga, 27, 0.12, True, None))      # check not shared
        labels = [label for ok, label in fit_checks(ganga, 60, 0.12, True, 0.74) if not ok]
        self.assertEqual(labels, ["Lends for up to 45 days"])

    def test_split_fills_cheapest_first(self):
        lenders = [self.net.lender("pennar"), self.net.lender("ganga")]
        parts = split_amount(3_160_000, lenders)
        self.assertEqual([(x.lid, p) for x, p in parts], [("ganga", 2_000_000), ("pennar", 1_160_000)])
        parts = split_amount(5_000_000, lenders)
        self.assertEqual(sum(p for _, p in parts), 3_500_000)    # not enough between them
        self.assertEqual(split_amount(0, lenders), [])

    def test_request_accept_reduces_available(self):
        r = self.net.send(lender_id="ganga", amount=1_500_000, rate=0.12, **self.kw)
        self.assertEqual(r.status, WAITING)
        self.assertEqual(self.net.requests(lender_id="ganga")[0].rid, r.rid)
        done = self.net.respond(r.rid, "accept")
        self.assertEqual(done.status, ACCEPTED)
        self.assertEqual(done.agreed_rate, 0.12)
        self.assertEqual(self.net.lender("ganga").available, 500_000)
        with self.assertRaises(ValueError):
            self.net.respond(r.rid, "decline")                   # already answered

    def test_accept_more_than_available_is_refused(self):
        r = self.net.send(lender_id="palar", amount=2_000_000, rate=0.12, **self.kw)
        with self.assertRaises(ValueError):
            self.net.respond(r.rid, "accept")
        self.assertEqual(self.net.request(r.rid).status, WAITING)

    def test_counter_offer_flow(self):
        r = self.net.send(lender_id="pennar", amount=1_000_000, rate=0.12, **self.kw)
        with self.assertRaises(ValueError):
            self.net.respond(r.rid, "counter", counter_rate=0.11)   # must be higher
        c = self.net.respond(r.rid, "counter", counter_rate=0.13)
        self.assertEqual((c.status, c.rate_now), (COUNTERED, 0.13))
        self.assertGreater(c.interest_now, interest(1_000_000, 0.12, c.days))
        a = self.net.answer_counter(r.rid, True)
        self.assertEqual((a.status, a.agreed_rate), (ACCEPTED, 0.13))
        self.assertEqual(self.net.lender("pennar").available, 500_000)

    def test_decline_withdraw_and_reset(self):
        r1 = self.net.send(lender_id="pennar", amount=500_000, rate=0.12, **self.kw)
        r2 = self.net.send(lender_id="yamuna", amount=500_000, rate=0.14, **self.kw)
        self.assertEqual(self.net.respond(r1.rid, "decline").status, DECLINED)
        self.assertEqual(self.net.withdraw(r2.rid).status, WITHDRAWN)
        with self.assertRaises(ValueError):
            self.net.withdraw(r1.rid)
        v = self.net.version
        self.net.set_conditions("pennar", min_rate=0.2)
        self.assertGreater(self.net.version, v)
        with self.assertRaises(ValueError):
            self.net.set_conditions("pennar", company="Renamed")
        self.net.reset()
        self.assertEqual(self.net.requests(), [])
        self.assertEqual(self.net.lender("pennar"), DEMO_LENDERS[1])
        r3 = self.net.send(lender_id="pennar", amount=500_000, rate=0.12, **self.kw)
        self.assertNotIn(r3.rid, {r1.rid, r2.rid})               # ids are never reused after a reset

    def test_concurrent_sends_get_unique_ids(self):
        def worker():
            for _ in range(50):
                self.net.send(lender_id="yamuna", amount=1_000, rate=0.14, **self.kw)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [r.rid for r in self.net.requests()]
        self.assertEqual(len(ids), 400)
        self.assertEqual(len(set(ids)), 400)

    def test_term_sheet_has_the_terms(self):
        sec = ({"invoice_id": "INV-1", "customer": "Kalinga Chassis Ltd", "amount": 600_000.0,
                "due_date": pd.Timestamp("2026-01-15"), "p_paid": 0.9},)
        r = self.net.send(lender_id="ganga", amount=500_000, rate=0.12, security=sec,
                          security_total=600_000.0, security_cover=0.95, **self.kw)
        r = self.net.respond(r.rid, "accept")
        text = term_sheet(r, self.net.lender("ganga"))
        for part in ("Ganga Castings Pvt Ltd", "Rs 5.00 lakh", "12.00% a year", "02 Feb 2026", "INV-1",
                     "Sec 186", "74%"):
            self.assertIn(part, text)


class TestCalendar(unittest.TestCase):

    def test_ics_has_one_all_day_event_with_reminder_per_date(self):
        ics = calendar_ics([(pd.Timestamp("2026-01-06"), "Receive from Ganga", "₹20.00 L, 12%; by RTGS"),
                            (pd.Timestamp("2026-02-02"), "Repay Ganga", "₹20.18 L")], "Runway: loan plan")
        self.assertTrue(ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n"))
        self.assertEqual(ics.count("BEGIN:VEVENT"), 2)
        self.assertEqual(ics.count("BEGIN:VALARM"), 2)
        self.assertIn("DTSTART;VALUE=DATE:20260106", ics)
        self.assertIn("DTEND;VALUE=DATE:20260203", ics)
        self.assertIn("SUMMARY:Repay Ganga", ics)
        self.assertIn("12%\\; by RTGS", ics)                     # special characters escaped


if __name__ == "__main__":
    unittest.main()
