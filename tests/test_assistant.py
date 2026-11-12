"""
Tests for the legal assistant (assistant/): interest maths, retrieval and the offline answer.
No network and no API key needed.

Run: python -m unittest tests.test_assistant -v
"""
import unittest

import pandas as pd

from assistant import BusinessContext, answer_offline
from assistant.interest import interest_for_invoices, msmed_interest
from assistant.legal_corpus import MSMED_RATE, PASSAGES
from assistant.retriever import LegalRetriever
from data.generator import generate_msme_ledger


class TestInterest(unittest.TestCase):

    def test_one_month_is_monthly_rate(self):
        got = msmed_interest(100_000, 365 / 12)
        self.assertAlmostEqual(got, 100_000 * MSMED_RATE / 12, delta=5)

    def test_compounds_monthly(self):
        twelve_months = msmed_interest(100_000, 365)
        self.assertAlmostEqual(twelve_months, 100_000 * ((1 + MSMED_RATE / 12) ** 12 - 1), delta=1)
        self.assertGreater(twelve_months, 100_000 * MSMED_RATE)

    def test_no_interest_before_due(self):
        self.assertEqual(msmed_interest(100_000, 0), 0.0)
        self.assertEqual(msmed_interest(100_000, -5), 0.0)

    def test_credit_period_capped_at_45_days(self):
        inv = pd.DataFrame([{
            "invoice_id": "X1", "customer_name": "Test Co", "amount": 100_000.0,
            "issue_date": pd.Timestamp("2025-01-01"), "payment_terms_days": 60,
            "is_disputed": False, "actual_payment_date": pd.NaT,
        }])
        (line,) = interest_for_invoices(inv, pd.Timestamp("2025-03-17"))
        self.assertEqual(line.legal_due_date, pd.Timestamp("2025-02-15"))   # 45 days, not 60
        self.assertTrue(line.terms_over_limit)
        self.assertEqual(line.days_late, 30)

    def test_paid_invoices_are_skipped(self):
        inv = pd.DataFrame([{
            "invoice_id": "X2", "customer_name": "Test Co", "amount": 50_000.0,
            "issue_date": pd.Timestamp("2025-01-01"), "payment_terms_days": 30,
            "is_disputed": False, "actual_payment_date": pd.Timestamp("2025-01-20"),
        }])
        self.assertEqual(interest_for_invoices(inv, pd.Timestamp("2025-06-01")), [])


class TestRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = LegalRetriever()

    def top(self, q):
        return [p.cite for p, _ in self.r.search(q, k=3)]

    def test_interest_question_finds_section_16(self):
        self.assertIn("Sec 16", self.top("can I charge interest on late payment"))

    def test_hinglish_word_for_interest(self):
        self.assertIn("Sec 16", self.top("byaj kitna milega"))

    def test_hindi_and_telugu_words(self):
        self.assertIn("Sec 16", self.top("क्या मैं ब्याज ले सकता हूँ?"))
        hits = self.top("देर से भुगतान करने वाले ग्राहक की शिकायत कैसे करूँ?")
        self.assertTrue({"Sec 18", "MSME Samadhaan"} & set(hits))
        self.assertIn("Sec 16", self.top("వడ్డీ వసూలు చేయవచ్చా?"))

    def test_complaint_question(self):
        hits = self.top("how do I file a complaint against a buyer")
        self.assertTrue({"Sec 18", "MSME Samadhaan"} & set(hits))

    def test_credit_limit_question(self):
        self.assertEqual(self.top("maximum credit period")[0], "Sec 15")

    def test_off_topic_returns_nothing(self):
        self.assertEqual(self.top("what is the weather today"), [])

    def test_statute_text_is_unchanged(self):
        s16 = next(p for p in PASSAGES if p.cite == "Sec 16")
        self.assertIn("compound interest with monthly rests", s16.text)
        self.assertIn("three times of the bank rate notified by the Reserve Bank", s16.text)


class TestOfflineAnswer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inv, _, _ = generate_msme_ledger(seed=42)
        as_of = pd.Timestamp(inv.attrs["as_of_date"])
        cls.ctx = BusinessContext("Demo Co", "Owner", as_of, 12.5e7, inv, 0.89, "Jan 07 – Jan 10")

    def test_named_customer_gets_their_invoices(self):
        ans, passages, facts = answer_offline("Can I legally charge Vindhya interest?", self.ctx)
        self.assertEqual(facts.customers, ["Vindhya Engineering Pvt Ltd"])
        self.assertIn("INV-2025-0240", ans)
        self.assertIn("[Sec 16", ans)
        self.assertIn("disputed", ans)
        cites = {p.cite for p in passages}
        self.assertTrue({"Sec 15", "Sec 16", "Sec 18", "Sec 2(n)"} <= cites)

    def test_every_cited_label_was_retrieved(self):
        for q in ("Can I legally charge Vindhya interest?", "How do I file a complaint?",
                  "Does my buyer lose a tax benefit if they pay late?", "Am I covered by the MSMED Act?"):
            ans, passages, _ = answer_offline(q, self.ctx)
            labels = {p.cite for p in passages}
            for bracket in pd.Series([ans]).str.findall(r"\[([^\]]+)\]")[0]:
                for label in (b.strip() for b in bracket.split(",")):
                    self.assertIn(label, labels, f"{q!r} cites [{label}] that was not retrieved")

    def test_unknown_topic_says_so(self):
        ans, _, _ = answer_offline("Who won the cricket match?", self.ctx)
        self.assertIn("could not find", ans)


if __name__ == "__main__":
    unittest.main()
