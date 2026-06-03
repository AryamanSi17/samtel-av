"""
Accuracy comparison: Tesseract vs EasyOCR on scanned PDFs.
Ground truth is built from actual values visible in the current PDFs.
"""

import io, re
import numpy as np
import fitz
from PIL import Image
import pytesseract
import easyocr
import pdfplumber

# ── Ground truth (from actual PDF content) ────────────────────────────────────
# Values are chosen to be clearly present in the documents.
# Partial GSTINs used (first 12 chars) to tolerate trailing OCR noise.

GROUND_TRUTH = {
    "INDIGIRD": {
        "invoice_number":  "AR1/2526/7284",
        "invoice_date":    "31-Mar-2026",
        "vendor_name":     "INDIGRID",
        "bill_to_gstin":   "06AACCR3527Q",   # partial
        "po_number":       "SADP25264100339",
        "ack_number":      "13262641806666",
        "payment_terms":   "Net-30",
        "place_of_supply": "Haryana",
        "hsn_sac":         "FC00833",
    },
    "LAPP": {
        "invoice_number":  "KA2627008229",
        "invoice_date":    "20.05.2026",
        "vendor_name":     "LAPP INDIA",
        "bill_to_gstin":   "06AACCR3527Q",
        "po_number":       "SADP25264100491",
        "hsn_sac":         "8489",
        "quantity":        "50",
        "payment_terms":   "Proforma",
    },
    "LPS": {
        "invoice_number":  "G100029587HR",
        "vendor_name":     "LPS BOSSARD",
        "bill_to_gstin":   "06AACCR3527Q",
        "hsn_sac":         "73181500",
        "customer_ref":    "S00029829",
        "description":     "Hex socket head cap screws",
        "quantity":        "400",
    },
    "RAMESHWARAM": {
        "invoice_number":  "RE/26-27/0060",
        "invoice_date":    "27-May-26",
        "vendor_gstin":    "07AATPS5809C1ZD",
        "bill_to_gstin":   "06AACCR3527Q",
        "total":           "7,911.90",
        "igst":            "1,206.90",
        "taxable_total":   "6,255.00",
        "hsn_sac":         "32141000",
    },
    "AGGRESSIVE": {
        "invoice_number":  "2026-27/J000111",
        "invoice_date":    "29-Apr-26",
        "vendor_gstin":    "06BAAECA8545M1ZB",
        "bill_to_gstin":   "06BAAKCS5764L1ZQ",
        "hsn_sac":         "998898",
        "quantity":        "2.00",
        "rate":            "1,092.72",
        "description":     "PCBA Manufacturing",
    },
    "GEM": {
        "invoice_date":    "29-Apr-2026",
        "vendor_gstin":    "07AAGCG8384A1ZL",
        "bill_to_gstin":   "09AACCR3527Q",   # partial
        "sac_code":        "998599",
        "taxable_amount":  "21,033.31",
        "ack_number":      "172620097692539",
        "gross_total":     "35,998.64",
    },
    "REIMBURSEMENT": {
        "section_a":        "BOARDING",
        "section_b":        "PERSONAL ALLOWANCE",
        "section_c":        "Conveyance",
        "budget_code":      "BUDGET CODE",
        "signature_line":   "Checked by",
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
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_scanned(path: str) -> bool:
    with open(path, "rb") as f:
        data = f.read()
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                return False
    return True


def ocr_tesseract(path: str, dpi: int = 300) -> str:
    with open(path, "rb") as f:
        data = f.read()
    doc = fitz.open(stream=data, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    parts = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        parts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(parts)


_reader = None

def ocr_easyocr(path: str, dpi: int = 300) -> str:
    global _reader
    if _reader is None:
        print("  [EasyOCR] Loading model...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    with open(path, "rb") as f:
        data = f.read()
    doc = fitz.open(stream=data, filetype="pdf")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    parts = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        result = _reader.readtext(np.array(img), detail=0, paragraph=False)
        parts.append(" ".join(result))
    doc.close()
    return "\n\n".join(parts)


def _norm(s: str) -> str:
    """Normalise for fuzzy comparison: uppercase, strip spaces/punctuation."""
    return re.sub(r'[\s\-/.,:()\']', '', s.upper())


def score(ocr_text: str, gt: dict) -> tuple[int, int, list[str]]:
    """Return (hits, total, missed_keys)."""
    nt = _norm(ocr_text)
    hits, missed = 0, []
    for key, val in gt.items():
        nv = _norm(str(val))
        if nv in nt:
            hits += 1
        else:
            missed.append(f"{key}={val!r}")
    return hits, len(gt), missed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*78}")
    print(f"{'Invoice':<22}  {'Tesseract':>14}  {'EasyOCR':>14}  {'Δ':>4}")
    print(f"{'-'*78}")

    total_t = total_e = total_fields = 0
    all_missed = {}

    for name, rel_path in PDF_MAP.items():
        path = f"/Users/aryaman/codebuggedprojects/samtel-av/{rel_path}"
        gt = GROUND_TRUTH[name]

        if not is_scanned(path):
            print(f"  {name:<20}  digital PDF — skipped")
            continue

        print(f"  Scanning {name}...", end="", flush=True)
        t_text = ocr_tesseract(path)
        e_text = ocr_easyocr(path)

        t_hits, fields, t_missed = score(t_text, gt)
        e_hits, _,      e_missed = score(e_text, gt)

        total_t      += t_hits
        total_e      += e_hits
        total_fields += fields

        delta = e_hits - t_hits
        ds = (f"+{delta}" if delta > 0 else str(delta)) if delta != 0 else " ="
        print(f"\r  {name:<20}  {t_hits}/{fields} ({100*t_hits//fields:3d}%)  "
              f"{e_hits}/{fields} ({100*e_hits//fields:3d}%)  {ds:>4}")

        if t_missed or e_missed:
            all_missed[name] = {"tesseract": t_missed, "easyocr": e_missed}

    print(f"{'='*78}")
    pct_t = 100 * total_t // total_fields
    pct_e = 100 * total_e // total_fields
    d = total_e - total_t
    ds = f"+{d}" if d > 0 else str(d)
    print(f"  {'OVERALL':<20}  {total_t}/{total_fields} ({pct_t:3d}%)  "
          f"{total_e}/{total_fields} ({pct_e:3d}%)  {ds:>4}")
    print(f"{'='*78}\n")

    print("── Missed fields ──────────────────────────────────────────────────────────")
    for name, m in all_missed.items():
        print(f"\n  {name}:")
        t_only = set(m["tesseract"]) - set(m["easyocr"])
        e_only = set(m["easyocr"])  - set(m["tesseract"])
        both   = set(m["tesseract"]) & set(m["easyocr"])
        if t_only:
            print(f"    Tesseract missed only: {', '.join(sorted(t_only))}")
        if e_only:
            print(f"    EasyOCR   missed only: {', '.join(sorted(e_only))}")
        if both:
            print(f"    Both missed:           {', '.join(sorted(both))}")


if __name__ == "__main__":
    main()
