import os
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from groq import Groq

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

load_dotenv()

app = FastAPI(title="Samtel Avionics Invoice Processor")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment or .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

ALLOWED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "image/webp",
}

# ── Invoice type classifier ────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are an invoice classifier for Samtel Avionics Limited, an avionics company in India.

STEP 1 — Find the ISSUER (the company that printed/sent the invoice):
- Look for the company name at the very top of the document, or in the "From" / "Supplier" / "Vendor" field.
- Samtel Avionics / Samtel HAL Display Systems appearing as ISSUER → SALE_DOMESTIC or SALE_EXPORT.
- Any other Indian company appearing as ISSUER, with Samtel as the buyer/billed-to → PURCHASE_DOMESTIC or SERVICES.
- UPS / DHL / FedEx as ISSUER with IEC code / foreign shipment → IMPORT.
- Government e-Marketplace (GEM) as ISSUER → SERVICES.
- Reimbursement/expense form with no issuer company → EMPLOYEE_REIMBURSEMENT.

STEP 2 — Pick exactly one code:

PURCHASE_DOMESTIC   - A non-Samtel Indian vendor issued the invoice; Samtel is the buyer. Goods/materials purchased.
IMPORT              - UPS/DHL/courier issued it; involves foreign shipment, IEC code, duty, foreign currency.
SALE_DOMESTIC       - Samtel Avionics (or Samtel HAL) issued the invoice; Indian buyer.
SALE_EXPORT         - Samtel issued the invoice; buyer is outside India (foreign address, no GST, SWIFT).
SERVICES            - A service/job-work provider OR GEM issued the invoice for services rendered to Samtel.
                      STRONG signals: document title says "JOB WORK INVOICE", HSN/SAC code starts with 99 (e.g. 998898),
                      description contains words like Manufacturing, Assembly, PCBA, Processing, Job Work, GEM, Transaction charges.
EMPLOYEE_REIMBURSEMENT - Internal employee expense reimbursement form.

Return ONLY the type code, nothing else.

Invoice text:
{invoice_text}"""

# ── Type-specific extraction prompts ──────────────────────────────────────────

PROMPTS = {

"PURCHASE_DOMESTIC": """Extract ALL fields from this domestic purchase invoice and return valid JSON only.

Fields:
- invoice_number
- invoice_date
- irn_number
- ack_number
- ack_date
- eway_bill_number
- vendor_name
- vendor_gstin
- vendor_pan
- vendor_address
- vendor_state
- vendor_state_code
- bill_to_name
- bill_to_address
- bill_to_gstin
- ship_to_name
- ship_to_address
- ship_to_gstin
- po_number
- po_date
- payment_terms
- line_items: array of {{ sno, material_code, description, hsn_sac, quantity, uom, rate, total, discount_pct, taxable_value, cgst_pct, cgst_amt, sgst_pct, sgst_amt, igst_pct, igst_amt }}
- taxable_total
- total_cgst
- total_sgst
- total_igst
- total_gst
- grand_total
- amount_in_words
- currency
- vendor_bank: {{ beneficiary, bank_name, account_no, ifsc, swift }}

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",

"IMPORT": """Extract ALL fields from this import/courier invoice and return valid JSON only.

Fields:
- invoice_subtype  (BILL_OF_SUPPLY or TAX_INVOICE)
- courier_company
- account_number
- customer_iec_code
- customer_pan
- customer_gstin
- invoice_number
- invoice_date
- irn_number
- ack_number
- ack_date
- place_of_supply
- shipment: {{ import_date, tracking_number, reference_number, shipment_number, service_type, packages, bill_type, weight_kg }}
- goods: {{ description, value, original_currency, customs_number, exchange_rate_to_inr }}
- shipper_name
- shipper_location
- hsn_code
- duty_amount_inr
- disbursement_fee_inr
- sgst_pct
- sgst_amt
- cgst_pct
- cgst_amt
- igst_pct
- igst_amt
- total_inr
- courier_bank: {{ bank_name, account_number, ifsc, swift }}

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",

"SALE_DOMESTIC": """Extract ALL fields from this domestic sales invoice (Samtel Avionics is the SELLER) and return valid JSON only.

Fields:
- invoice_number
- invoice_date
- irn_number
- ack_number
- ack_date
- eway_bill_number
- seller_name
- seller_address
- seller_gstin
- seller_pan
- seller_cin
- seller_state
- seller_state_code
- buyer_name  (bill-to)
- buyer_address
- buyer_gstin
- buyer_state
- buyer_state_code
- consignee_name  (ship-to, may differ from buyer)
- consignee_address
- consignee_gstin
- consignee_state
- delivery_note_number
- samtel_reference
- buyer_order_number
- buyer_order_date
- payment_mode
- insurance_policy
- dispatched_through
- destination
- line_items: array of {{ sno, description, part_number, serial_numbers, hsn_sac, quantity, uom, rate, amount }}
- tax_base_amount
- igst_pct
- igst_amount
- cgst_pct
- cgst_amount
- sgst_pct
- sgst_amount
- total_amount
- amount_in_words
- payment_terms
- currency
- seller_bank: {{ ac_holder, bank_name, account_no, branch_ifsc, swift_code }}

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",

"SALE_EXPORT": """Extract ALL fields from this export sales invoice (Samtel Avionics is the SELLER, buyer is a foreign entity) and return valid JSON only.

Fields:
- invoice_number
- invoice_date
- irn_number  (may be blank for exports)
- ack_number
- ack_date
- eway_bill_number
- seller_name
- seller_address
- seller_gstin
- seller_pan
- seller_cin
- buyer_name
- buyer_address
- buyer_country
- buyer_gstin  (null for foreign buyers)
- consignee_name
- consignee_address
- consignee_country
- delivery_note_number
- samtel_reference
- buyer_order_number
- buyer_order_date
- payment_mode
- dispatched_through
- destination_country
- line_items: array of {{ sno, description, part_number, serial_numbers, hsn_sac, quantity, uom, rate_inr, amount_inr }}
- tax_base_amount_inr
- gst_applicable  (false for exports)
- igst_amount  (should be 0 for exports)
- total_amount_inr
- amount_in_words
- payment_terms
- currency
- seller_bank: {{ ac_holder, bank_name, account_no, branch_ifsc, swift_code }}

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",

"SERVICES": """Extract ALL fields from this service/job-work/GEM invoice and return valid JSON only.

First identify the sub-type: JOB_WORK, GEM, or GENERAL_SERVICE.

Fields:
- invoice_subtype  (JOB_WORK / GEM / GENERAL_SERVICE)
- invoice_number
- invoice_date
- irn_number
- ack_number
- ack_date
- service_provider_name
- service_provider_gstin
- service_provider_address
- service_provider_state
- bill_to_name
- bill_to_address
- bill_to_gstin
- bill_to_state
- place_of_supply
- rcm_applicable
- payment_terms
- line_items: array of {{ sno, description, order_number, hsn_sac, quantity, uom, rate, taxable_amount, gst_pct, gst_amount }}
- taxable_total
- cgst_pct
- cgst_amt
- sgst_pct
- sgst_amt
- igst_pct
- igst_amt
- total_gst
- gross_total
- tds_pct  (if TDS deducted, e.g. GEM invoices)
- tds_amount
- net_payable_after_tds
- amount_in_words
- currency
- nrgp_number  (for job-work)
- eway_bill_number
- challan_reference
- bank_details: {{ beneficiary, bank_name, account_no, ifsc }}

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",

"EMPLOYEE_REIMBURSEMENT": """Extract ALL fields from this employee reimbursement form and return valid JSON only.

Fields:
- employee_name
- employee_sap_code
- department
- budget_code
- advance_amount
- section_a_travel: {{ boarding, lodging, personal_allowance, total }}
- section_b_allowances: {{ total }}
- total_a_plus_b
- section_c_expenses: {{ conveyance, telephone, postage, stationery, misc_expenses, ticket_charges, business_promotion, total }}
- grand_total
- less_air_ticket_by_company
- net_payable_to_employee
- checked_by
- passed_by
- head_of_department
- currency

Return null for any missing field. Return ONLY valid JSON.

Invoice text:
{invoice_text}""",
}

# ── Text extraction ────────────────────────────────────────────────────────────

def _ocr_pdf_with_pymupdf(file_bytes: bytes, dpi: int = 300) -> str:
    """Render each page with PyMuPDF and OCR with Tesseract."""
    if not TESSERACT_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="This PDF appears to be scanned. OCR requires Tesseract: brew install tesseract",
        )
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 dpi is PyMuPDF default
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        parts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n\n".join(parts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    # 1. Try native text extraction via pdfplumber (digital PDFs — lossless)
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    if text_parts:
        return "\n\n".join(text_parts)

    # 2. Scanned PDF — render with PyMuPDF then OCR
    return _ocr_pdf_with_pymupdf(file_bytes, dpi=300)


def extract_text_from_image(file_bytes: bytes) -> str:
    if not TESSERACT_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Image OCR requires Tesseract. Install it with: brew install tesseract",
        )
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm(prompt: str, max_tokens: int = 256) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_json(content: str) -> dict:
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content, "parse_error": "Could not parse JSON from model response"}


def _keyword_classify(text: str) -> str | None:
    """Fast keyword pre-classifier — returns a type if unambiguous, else None."""
    import re
    t = text.upper()

    # 1. Services / job-work (check early — strong unambiguous signals)
    if "JOB WORK INVOICE" in t or "JOB WORK" in t:
        return "SERVICES"
    if "GOVERNMENT E-MARKETPLACE" in t or "GOVERNMENT EMARKETPLACE" in t:
        return "SERVICES"
    if "GEM-20" in t and ("TRANSACTION CHARGES" in t or "MILESTONE" in t):
        return "SERVICES"
    if re.search(r'\b998[0-9]\d\d\b', t):  # SAC codes 998xxx = services
        return "SERVICES"

    # 2. Employee reimbursement
    if "REIMBURSEMENT" in t and ("EMPLOYEE" in t or "SAP NO" in t or "DEPARTMENT" in t):
        return "EMPLOYEE_REIMBURSEMENT"

    # 3. Import — courier + foreign shipment (IEC code is a very strong signal)
    if re.search(r'IEC\s*(NO|CODE|NUMBER)?[\s:]*\d', t):
        return "IMPORT"
    if ("UPS EXPRESS" in t or "DHL EXPRESS" in t or "FEDEX" in t) and ("DUTY" in t or "USD" in t):
        return "IMPORT"

    # 4. Determine if Samtel is the ISSUER (top ~400 chars) or the BUYER
    # Strip everything from the first address-block label onward so we only see the issuer header
    address_cut = re.search(r'\b(SHIP\s*TO|BILL\s*TO|CONSIGNEE|BUYER|BILLED\s*TO|RECEIVER)\b', t)
    header_zone = t[:address_cut.start()] if address_cut else t[:400]
    samtel_is_issuer = "SAMTEL AVIONICS" in header_zone or "SAMTEL HAL DISPLAY" in header_zone

    if samtel_is_issuer:
        # 5. Export: buyer has no GSTIN (foreign entity)
        # Look for the buyer/bill-to GSTIN block — if it's blank after the buyer address, it's export
        buyer_block = re.search(r'(BUYER|BILL\s*TO).{0,600}GSTIN[/\s]*UIN\s*:\s*(\S*)', t, re.DOTALL)
        if buyer_block:
            gstin_val = buyer_block.group(2).strip()
            # Foreign buyers have empty or absent GSTIN
            if not gstin_val or len(gstin_val) < 10:
                return "SALE_EXPORT"
        return "SALE_DOMESTIC"

    # 6. If Samtel is NOT the issuer → purchase or services (let LLM handle ambiguous cases)
    return None


def classify_invoice(raw_text: str) -> str:
    # Try fast keyword match first
    fast = _keyword_classify(raw_text)
    if fast:
        return fast

    # Fall back to LLM
    result = _llm(CLASSIFY_PROMPT.format(invoice_text=raw_text[:4000]), max_tokens=16)
    valid = {"PURCHASE_DOMESTIC", "IMPORT", "SALE_DOMESTIC", "SALE_EXPORT", "SERVICES", "EMPLOYEE_REIMBURSEMENT"}
    for v in valid:
        if v in result.upper():
            return v
    return "PURCHASE_DOMESTIC"  # safe fallback


def extract_invoice_data(invoice_type: str, raw_text: str) -> dict:
    prompt = PROMPTS[invoice_type].format(invoice_text=raw_text)
    content = _llm(prompt, max_tokens=3000)
    return _parse_json(content)


# ── API ────────────────────────────────────────────────────────────────────────

@app.post("/api/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a PDF or image file.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file.content_type == "application/pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_image(file_bytes)

    if not raw_text or len(raw_text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Could not extract readable text. Ensure the file is not a low-resolution scan.",
        )

    invoice_type = classify_invoice(raw_text)
    invoice_data = extract_invoice_data(invoice_type, raw_text)

    return {
        "status": "success",
        "filename": file.filename,
        "invoice_type": invoice_type,
        "raw_text_preview": raw_text[:500] + ("..." if len(raw_text) > 500 else ""),
        "invoice_data": invoice_data,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


app.mount("/static", StaticFiles(directory="static"), name="static")
