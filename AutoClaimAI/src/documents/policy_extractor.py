import re
import fitz  # PyMuPDF


def extract_policy_text(pdf_bytes: bytes) -> dict:
    """Extract text and identify possible policy details from PDF bytes."""
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "fields": {},
            "message": f"Could not open the PDF: {exc}",
        }

    pages = []
    try:
        for page in document:
            pages.append(page.get_text())
    finally:
        document.close()

    text = "\n".join(pages).strip()

    patterns = {
        "policy_number": r"(?i)\bpolicy\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9/\-]{4,})",
        "deductible": r"(?i)\b(?:deductible|excess)\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)",
        "policy_period": r"(?i)\bpolicy\s*period\s*[:\-]?\s*([^\n]+)",
    }

    fields = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[field] = match.group(1).strip() if match else None

    return {
        "success": bool(text),
        "text": text,
        "fields": fields,
        "message": (
            "Text extracted. Verify the extracted fields against the original policy."
            if text
            else "No selectable text found. The PDF may be scanned and require OCR."
        ),
    }