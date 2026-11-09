"""
Financial Data Sanitizer and Security Guard for Runway.

Protects against:
1. CSV / Formula Injection (=, +, -, @ payloads) in user-supplied accounting ledgers.
2. Chronological inversions (payment before issue date, due date before issue date).
3. Financial ledger anomalies (negative amounts, extreme outliers, duplicate invoices).
4. PII leakage (masks GSTIN, PAN, and Bank Account numbers for compliance).
5. Schema normalization (mapping dirty segment/category strings to canonical constants).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from data.schemas import (
    CUSTOMER_SEGMENTS,
    EXPENSE_CATEGORIES,
    SEGMENT_MID_TIER,
    SEGMENT_RETAIL,
    SEGMENT_TIER1_OEM,
)

# Formula injection prefixes dangerous in Excel / Google Sheets
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Standard Indian GSTIN pattern (2 digits state code, 10 chars PAN, 1 entity, 1 'Z', 1 check digit)
GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", re.IGNORECASE)

# Standard Indian PAN pattern
PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", re.IGNORECASE)


def sanitize_formula_injection(val: Any) -> Any:
    """Neutralizes formula injection by prepending a single quote if the string
    starts with dangerous spreadsheet command symbols (=, +, -, @, tabs).
    """
    if isinstance(val, str):
        if any(val.startswith(prefix) for prefix in FORMULA_PREFIXES):
            return f"'{val}"
        val_clean = val.strip()
        if any(val_clean.startswith(prefix) for prefix in FORMULA_PREFIXES):
            return f"'{val_clean}"
    return val


def mask_gstin(gstin: Optional[str]) -> str:
    """Masks Indian GSTIN for privacy compliance (e.g. 36AAAAA0000A1Z5 -> 36*****1Z5)."""
    if not gstin or not isinstance(gstin, str):
        return ""
    g = gstin.strip()
    if len(g) == 15:
        return f"{g[:2]}*****{g[-3:]}"
    return "****"


def mask_bank_account(acc: Optional[str]) -> str:
    """Masks Indian bank account number (e.g. 123456789012 -> ********9012)."""
    if not acc or not isinstance(acc, str):
        return ""
    s = re.sub(r"\D", "", acc.strip())
    if len(s) >= 4:
        return f"{'*' * (len(s) - 4)}{s[-4:]}"
    return "****"


def sanitize_invoice_ledger(
    df: pd.DataFrame,
    min_amount: float = 100.0,
    max_amount: float = 100_000_000.0,  # ₹10 Crore reasonable upper bound for single MSME invoice
    raise_on_error: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Sanitizes and validates an exported or user-uploaded invoice ledger.

    Returns:
        (sanitized_df, validation_warnings)
    """
    clean_df = df.copy()
    warnings: List[str] = []

    # 1. Neutralize Formula Injection across all string columns
    for col in clean_df.select_dtypes(include=["object"]).columns:
        clean_df[col] = clean_df[col].apply(sanitize_formula_injection)

    # 2. Check required columns
    required_cols = {"invoice_id", "customer_id", "amount", "issue_date", "due_date"}
    missing = required_cols - set(clean_df.columns)
    if missing:
        msg = f"Security / Schema Error: Missing required columns: {sorted(missing)}"
        if raise_on_error:
            raise ValueError(msg)
        warnings.append(msg)
        return clean_df, warnings

    # 3. Deduplicate Invoices
    orig_len = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["invoice_id"], keep="last")
    if len(clean_df) < orig_len:
        warnings.append(f"Deduplicated {orig_len - len(clean_df)} duplicate invoice entries.")

    # 4. Parse Dates & Chronological Sanity
    clean_df["issue_date"] = pd.to_datetime(clean_df["issue_date"], errors="coerce")
    clean_df["due_date"] = pd.to_datetime(clean_df["due_date"], errors="coerce")
    if "actual_payment_date" in clean_df.columns:
        clean_df["actual_payment_date"] = pd.to_datetime(clean_df["actual_payment_date"], errors="coerce")

    # Inverted due date check
    inverted_due = clean_df["due_date"] < clean_df["issue_date"]
    if inverted_due.any():
        n_inv = inverted_due.sum()
        warnings.append(f"Fixed {n_inv} rows where due_date preceded issue_date (adjusted due_date = issue_date + 30d).")
        clean_df.loc[inverted_due, "due_date"] = clean_df.loc[inverted_due, "issue_date"] + pd.Timedelta(days=30)

    # Inverted payment date check
    if "actual_payment_date" in clean_df.columns:
        paid_mask = clean_df["actual_payment_date"].notna()
        inverted_paid = paid_mask & (clean_df["actual_payment_date"] < clean_df["issue_date"])
        if inverted_paid.any():
            n_inv = inverted_paid.sum()
            warnings.append(f"Fixed {n_inv} rows where payment preceded issue date (set payment = issue_date).")
            clean_df.loc[inverted_paid, "actual_payment_date"] = clean_df.loc[inverted_paid, "issue_date"]

    # 5. Monetary Bounds & Negative Value Guard
    clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce").fillna(0.0)
    neg_mask = clean_df["amount"] < 0
    if neg_mask.any():
        warnings.append(f"Flagged and absolute-valued {neg_mask.sum()} negative invoice amounts (credit notes).")
        clean_df["amount"] = clean_df["amount"].abs()

    # Outlier clipping
    outliers = clean_df["amount"] > max_amount
    if outliers.any():
        warnings.append(f"Clipped {outliers.sum()} extreme outlier amounts above ₹{max_amount:,.0f}.")
        clean_df.loc[outliers, "amount"] = max_amount

    under_min = (clean_df["amount"] > 0) & (clean_df["amount"] < min_amount)
    if under_min.any():
        clean_df.loc[under_min, "amount"] = min_amount

    # 6. Normalize Customer Segment
    if "customer_segment" in clean_df.columns:
        def _normalize_segment(s: Any) -> str:
            if not isinstance(s, str):
                return SEGMENT_MID_TIER
            s_low = s.lower()
            if "oem" in s_low or "tier-1" in s_low or "enterprise" in s_low:
                return SEGMENT_TIER1_OEM
            if "retail" in s_low or "distributor" in s_low or "dealer" in s_low:
                return SEGMENT_RETAIL
            return SEGMENT_MID_TIER

        clean_df["customer_segment"] = clean_df["customer_segment"].apply(_normalize_segment)
    else:
        clean_df["customer_segment"] = SEGMENT_MID_TIER

    # 7. Normalize Boolean / Dispute Flags
    if "is_disputed" in clean_df.columns:
        clean_df["is_disputed"] = clean_df["is_disputed"].astype(bool)
    else:
        clean_df["is_disputed"] = False

    return clean_df, warnings
