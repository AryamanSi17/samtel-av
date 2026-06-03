"""
End-to-end accuracy test: text extraction → classification → LLM field extraction.
Tests all 11 PDFs and scores each against ground truth.
"""

import sys, os, time, re as _re
sys.path.insert(0, os.path.dirname(__file__))

# Import the full pipeline from main.py
from main import (
    extract_text_from_pdf,
    classify_invoice,
    extract_invoice_data,
)


def _parse_wait(msg: str) -> float | None:
    """Parse Groq rate-limit message into seconds to wait. Handles Xh Ym Z.Ws and Ym Z.Ws."""
    m = _re.search(r'(\d+)h(\d+)m([\d.]+)s', msg)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = _re.search(r'(\d+)m([\d.]+)s', msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def _retry_llm(fn, *args, max_wait=10800, **kwargs):
    """Call fn(*args, **kwargs); if Groq 429, parse wait time and retry (up to max_wait secs)."""
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            wait = _parse_wait(msg)
            if "429" in msg and wait is not None:
                wait += 15  # small buffer
                if wait > max_wait:
                    raise
                print(f"\n    [rate limit] waiting {wait/60:.1f} min...", end="", flush=True)
                time.sleep(wait)
                print(" retrying...", end="", flush=True)
            else:
                raise

import re, json

# ── Ground truth ──────────────────────────────────────────────────────────────

GROUND_TRUTH = {
    # ── Scanned purchase invoices ─────────────────────────────────────────────
    "INDIGIRD": {
        "expected_type": "PURCHASE_DOMESTIC",
        "fields": {
            "invoice_number":  "AR1/2526/7284",
            "invoice_date":    "31-Mar-2026",
            "vendor_name":     "INDIGRID",
            "bill_to_gstin":   "06AACCR3527Q",
            "po_number":       "SADP25264100339",
            "ack_number":      "13262641806666",
            "payment_terms":   "Net-30",
        },
    },
    "LAPP": {
        "expected_type": "PURCHASE_DOMESTIC",
        "fields": {
            "invoice_number":  "KA2627008229",
            "invoice_date":    "20.05.2026",
            "vendor_name":     "LAPP INDIA",
            "bill_to_gstin":   "06AACCR3527Q",
            "po_number":       "SADP25264100491",
            "hsn_sac":         "8489",
        },
    },
    "LPS": {
        "expected_type": "PURCHASE_DOMESTIC",
        "fields": {
            "invoice_number":  "G100029587HR",
            "vendor_name":     "LPS BOSSARD",
            "bill_to_gstin":   "06AACCR3527Q",
            "hsn_sac":         "73181500",
            "po_number":       "S00029829",
        },
    },
    "RAMESHWARAM": {
        "expected_type": "PURCHASE_DOMESTIC",
        "fields": {
            "invoice_number":  "RE/26-27/0060",
            "invoice_date":    "27-May-26",
            "vendor_gstin":    "07AATPS5809C1ZD",
            "bill_to_gstin":   "06AACCR3527Q",
            "taxable_total":   "6255",
            "total_igst":      "1206",
            "grand_total":     "7911",
        },
    },
    # ── Scanned service invoices ──────────────────────────────────────────────
    "AGGRESSIVE": {
        "expected_type": "SERVICES",
        "fields": {
            "invoice_number":          "2026-27/J000111",
            "invoice_date":            "29-Apr-26",
            "service_provider_gstin":  "06BAAECA8545M1ZB",
            "bill_to_gstin":           "06BAAKCS5764L1ZQ",
            "hsn_sac":                 "998898",
            "description":             "PCBA",
        },
    },
    "GEM": {
        "expected_type": "SERVICES",
        "fields": {
            "invoice_date":            "29-Apr-2026",
            "service_provider_gstin":  "07AAGCG8384A1ZL",
            "bill_to_gstin":           "09AACCR3527Q",
            "hsn_sac":                 "998599",
            "taxable_total":           "21033",
            "gross_total":             "35998",
        },
    },
    # ── Scanned reimbursement ─────────────────────────────────────────────────
    "REIMBURSEMENT": {
        "expected_type": "EMPLOYEE_REIMBURSEMENT",
        "fields": {
            "section_a_travel":   "boarding",   # partial match - structural label
            "section_c_expenses": "conveyance",
        },
    },
    # ── Digital import invoice ────────────────────────────────────────────────
    "IMPORT": {
        "expected_type": "IMPORT",
        "fields": {
            "invoice_number":     "406500076092",
            "invoice_date":       "25 May 2026",
            "courier_company":    "UPS",
            "customer_iec_code":  "0501028226",
            "customer_gstin":     "06AACCR3527Q",
            "tracking_number":    "1ZG2B481",
            "duty_amount_inr":    "4295",
            "total_inr":          "4295",
        },
    },
    # ── Digital sale invoices ─────────────────────────────────────────────────
    "SALE_SIEMENS": {
        "expected_type": "SALE_DOMESTIC",
        "fields": {
            "invoice_number":   "SA/TI/20260007",
            "invoice_date":     "06-May-2026",
            "seller_gstin":     "06AACCR3527Q",
            "buyer_name":       "SIEMENS",
            "buyer_gstin":      "27AAACS0764L1Z6",
            "consignee_name":   "HORIZON MICROTECH",
            "hsn_sac":          "85371000",
            "total_amount":     "2525200",
            "igst_amount":      "385200",
        },
    },
    "SALE_DELTA": {
        "expected_type": "SALE_DOMESTIC",
        "fields": {
            "invoice_number":   "SA/TI/20260005",
            "invoice_date":     "30-Apr-2026",
            "seller_gstin":     "06AACCR3527Q",
            "buyer_name":       "Delta Engineering",
            "buyer_gstin":      "23AGRPJ5342D1ZK",
            "hsn_sac":          "85176290",
            "total_amount":     "201780",
            "igst_amount":      "30780",
        },
    },
    "SALE_SAAB": {
        "expected_type": "SALE_EXPORT",
        "fields": {
            "invoice_number":   "2026002",
            "invoice_date":     "20-May-2026",
            "seller_gstin":     "06AACCR3527Q",
            "buyer_name":       "Saab Grintek",
            "buyer_country":    "SOUTH AFRICA",
            "hsn_sac":          "88073000",
            "total_amount":     "1734896",
        },
    },
}

PDF_MAP = {
    "INDIGIRD":      "Automation_AI/PURCHASE/INDIGIRD TECHNOLOGY PVT LTD.pdf",
    "LAPP":          "Automation_AI/PURCHASE/LAPP INDIA PVT LTD.pdf",
    "LPS":           "Automation_AI/PURCHASE/LPS BOSSARD PVT LTD_PURCHASE.pdf",
    "RAMESHWARAM":   "Automation_AI/PURCHASE/RAMESHWARAM ENTERPRISES_PURCHASE.pdf",
    "AGGRESSIVE":    "Automation_AI/SERVICES/AGGRESSIVE JOB WORK.pdf",
    "GEM":           "Automation_AI/SERVICES/GEM TAX INVOICE.pdf",
    "REIMBURSEMENT": "Automation_AI/Employee reimbursement/Employee reimbursement.pdf",
    "IMPORT":        "Automation_AI/IMPORT/IMPORT_MOUSER ELECTRONICS,.pdf",
    "SALE_SIEMENS":  "Automation_AI/SALE DEMOSTIC AND EXPORT/SALE DEMOSTIC_007_SA TO SIEMENS_06-05-2026.pdf",
    "SALE_DELTA":    "Automation_AI/SALE DEMOSTIC AND EXPORT/SALE DEMOSTIC_SA_DETLA_0005.pdf",
    "SALE_SAAB":     "Automation_AI/SALE DEMOSTIC AND EXPORT/SALE EXPORT_002_SAAB_EXPORT_COMMERCIAL INVOICE & TAX INVOICE.pdf",
}

# ── Scoring ───────────────────────────────────────────────────────────────────

def _norm(s) -> str:
    if s is None:
        return ""
    return re.sub(r'[\s\-/.,:()\'"_]', '', str(s).upper())


def _flatten(obj, prefix="") -> dict:
    """Flatten nested JSON to dot-notation keys for scoring."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def score_extraction(extracted: dict, gt_fields: dict) -> tuple[int, int, list]:
    flat = _flatten(extracted)
    flat_norm = " ".join(_norm(v) for v in flat.values() if v)

    hits, missed = 0, []
    for key, expected in gt_fields.items():
        ne = _norm(str(expected))
        if ne and ne in flat_norm:
            hits += 1
        else:
            missed.append(f"{key}={expected!r}")
    return hits, len(gt_fields), missed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(__file__)
    total_type_correct = 0
    total_field_hits = 0
    total_fields = 0

    print(f"\n{'='*82}")
    print(f"{'Invoice':<18}  {'Expected Type':<24}  {'Got Type':<24}  {'Fields':>8}")
    print(f"{'-'*82}")

    all_missed = {}

    for name, rel_path in PDF_MAP.items():
        path = os.path.join(base, rel_path)
        gt = GROUND_TRUTH[name]
        expected_type = gt["expected_type"]
        gt_fields = gt["fields"]

        print(f"  Processing {name}...", end="", flush=True)

        try:
            with open(path, "rb") as f:
                raw = extract_text_from_pdf(f.read())

            invoice_type = _retry_llm(classify_invoice, raw)
            extracted    = _retry_llm(extract_invoice_data, invoice_type, raw)

            type_ok = invoice_type == expected_type
            hits, total, missed = score_extraction(extracted, gt_fields)

            total_type_correct += int(type_ok)
            total_field_hits   += hits
            total_fields       += total

            type_mark = "✓" if type_ok else "✗"
            pct = 100 * hits // total if total else 0
            print(f"\r  {name:<18}  {expected_type:<24}  "
                  f"{type_mark} {invoice_type:<22}  {hits}/{total} ({pct:3d}%)")

            if missed:
                all_missed[name] = missed

        except Exception as e:
            print(f"\r  {name:<18}  ERROR: {e}")

    n = len(PDF_MAP)
    print(f"{'='*82}")
    print(f"  Classification: {total_type_correct}/{n} correct  "
          f"({100*total_type_correct//n}%)")
    print(f"  Field accuracy: {total_field_hits}/{total_fields}  "
          f"({100*total_field_hits//total_fields if total_fields else 0}%)")
    print(f"{'='*82}\n")

    if all_missed:
        print("── Missed fields ─────────────────────────────────────────────────────────────")
        for name, missed in all_missed.items():
            print(f"  {name}: {', '.join(missed)}")
        print()


if __name__ == "__main__":
    main()
