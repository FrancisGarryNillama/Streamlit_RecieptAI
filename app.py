import base64
import datetime as dt
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from openai import OpenAI


DB_PATH = "receipts.db"


DOCUMENT_TYPES = [
    "official_receipt",
    "invoice",
    "sales_invoice",
    "delivery_receipt",
    "collection_receipt",
    "acknowledgment_receipt",
    "charge_invoice",
    "cash_invoice",
    "debit_memo",
    "credit_memo",
    "job_order",
    "purchase_order",
    "billing_statement",
    "statement_of_account",
    "unknown",
]

VAT_TYPES = ["vat", "non_vat", "zero_rated", "vat_exempt", "unknown"]

EXPENSE_CATEGORIES = [
    "office_supplies",
    "meals_entertainment",
    "transportation",
    "utilities",
    "communication",
    "professional_fees",
    "rent",
    "salaries",
    "repairs_maintenance",
    "taxes_licenses",
    "insurance",
    "advertising",
    "miscellaneous",
    "uncategorized",
]


EXTRACTION_SCHEMA = {
    "document_type": "|".join(DOCUMENT_TYPES),
    "vat_type": "|".join(VAT_TYPES),
    "expense_category": "|".join(EXPENSE_CATEGORIES),
    "business_name": "",
    "business_address": "",
    "tin": "",
    "receipt_number": "",
    "bir_permit_number": "",
    "expense_date": "YYYY-MM-DD or empty string",
    "description": "brief description of what was purchased",
    "buyer_name": "",
    "buyer_tin": "",
    "subtotal": 0.00,
    "vatable_sales": 0.00,
    "vat_exempt_sales": 0.00,
    "zero_rated_sales": 0.00,
    "vat_amount": 0.00,
    "total": 0.00,
}


@dataclass
class ReceiptRow:
    id: int
    business_name: str
    total: float
    expense_date: str
    tin: str
    receipt_number: str
    document_type: str
    description: str
    raw_ocr_json: str


def get_openai_api_key() -> Optional[str]:
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    return key or os.environ.get("OPENAI_API_KEY")


def get_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in Streamlit secrets or environment variables."
        )
    return OpenAI(api_key=api_key)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS receipts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          business_name TEXT,
          total REAL,
          expense_date TEXT,
          tin TEXT,
          receipt_number TEXT,
          document_type TEXT,
          description TEXT,
          raw_ocr_json TEXT
        )
        """
    )
    conn.commit()


def normalize_date_yyyy_mm_dd(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.date().isoformat() if isinstance(value, dt.datetime) else value.isoformat()
    s = str(value).strip()
    if not s:
        return ""
    try:
        return dt.date.fromisoformat(s).isoformat()
    except Exception:
        return s


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def insert_receipt(
    conn: sqlite3.Connection,
    *,
    business_name: str,
    total: float,
    expense_date: str,
    tin: str,
    receipt_number: str,
    document_type: str,
    description: str,
    raw_ocr_json: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO receipts
          (business_name, total, expense_date, tin, receipt_number, document_type, description, raw_ocr_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            business_name,
            float(total),
            expense_date,
            tin,
            receipt_number,
            document_type,
            description,
            raw_ocr_json,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def fetch_receipts(conn: sqlite3.Connection, limit: Optional[int] = None) -> List[ReceiptRow]:
    sql = """
      SELECT id, business_name, total, expense_date, tin, receipt_number, document_type, description, raw_ocr_json
      FROM receipts
      ORDER BY id DESC
    """
    params: Tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    rows = conn.execute(sql, params).fetchall()
    return [
        ReceiptRow(
            id=int(r["id"]),
            business_name=str(r["business_name"] or ""),
            total=float(r["total"] or 0.0),
            expense_date=str(r["expense_date"] or ""),
            tin=str(r["tin"] or ""),
            receipt_number=str(r["receipt_number"] or ""),
            document_type=str(r["document_type"] or ""),
            description=str(r["description"] or ""),
            raw_ocr_json=str(r["raw_ocr_json"] or ""),
        )
        for r in rows
    ]


def receipts_to_markdown_table(receipts: List[ReceiptRow]) -> str:
    headers = [
        "id",
        "business_name",
        "expense_date",
        "document_type",
        "receipt_number",
        "tin",
        "total",
        "description",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in receipts:
        cells = [
            str(r.id),
            (r.business_name or "").replace("\n", " ").strip(),
            (r.expense_date or "").replace("\n", " ").strip(),
            (r.document_type or "").replace("\n", " ").strip(),
            (r.receipt_number or "").replace("\n", " ").strip(),
            (r.tin or "").replace("\n", " ").strip(),
            f"{r.total:.2f}",
            (r.description or "").replace("\n", " ").strip(),
        ]
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def encode_image_to_data_url(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_receipt_json_from_image(client: OpenAI, uploaded_file) -> Dict[str, Any]:
    data_url = encode_image_to_data_url(uploaded_file)

    system_instructions = (
        "You are an OCR + data extraction assistant for Philippine receipts/invoices.\n"
        "Return ONLY valid JSON (no markdown, no code fences) that matches this exact schema.\n"
        "If a field is missing, use empty string for text fields and 0.00 for numeric fields.\n"
        "Allowed values:\n"
        f"- document_type: {', '.join(DOCUMENT_TYPES)}\n"
        f"- vat_type: {', '.join(VAT_TYPES)}\n"
        f"- expense_category: {', '.join(EXPENSE_CATEGORIES)}\n"
        "expense_date must be 'YYYY-MM-DD' or empty string.\n"
        "Numbers must be plain decimals (no currency symbols)."
    )

    user_prompt = (
        "Extract the receipt/invoice details from this image into JSON using this schema:\n"
        + json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False)
    )

    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ],
    )

    text = (resp.output_text or "").strip()
    if not text:
        raise ValueError("OpenAI returned an empty response.")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise

    if not isinstance(obj, dict):
        raise ValueError("OCR extraction did not return a JSON object.")
    return obj


def coerce_extracted_fields(extracted: Dict[str, Any]) -> Dict[str, Any]:
    document_type = str(extracted.get("document_type") or "unknown").strip() or "unknown"
    if document_type not in DOCUMENT_TYPES:
        document_type = "unknown"

    business_name = str(extracted.get("business_name") or "").strip()
    tin = str(extracted.get("tin") or "").strip()
    receipt_number = str(extracted.get("receipt_number") or "").strip()
    description = str(extracted.get("description") or "").strip()

    expense_date = normalize_date_yyyy_mm_dd(extracted.get("expense_date"))
    total = safe_float(extracted.get("total"))

    extracted = dict(extracted)
    extracted["document_type"] = document_type
    extracted["business_name"] = business_name
    extracted["tin"] = tin
    extracted["receipt_number"] = receipt_number
    extracted["description"] = description
    extracted["expense_date"] = expense_date if expense_date else ""
    extracted["total"] = total
    return extracted


def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_recent_uploads_sidebar(conn: sqlite3.Connection) -> None:
    st.sidebar.subheader("Recent Uploads")
    receipts = fetch_receipts(conn, limit=25)
    rows = [
        {
            "id": r.id,
            "business_name": r.business_name,
            "expense_date": r.expense_date,
            "document_type": r.document_type,
            "receipt_number": r.receipt_number,
            "tin": r.tin,
            "total": r.total,
            "description": r.description,
        }
        for r in receipts
    ]
    if pd is not None:
        st.sidebar.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.sidebar.dataframe(rows, use_container_width=True)


def render_uploader_sidebar(conn: sqlite3.Connection) -> None:
    st.sidebar.subheader("OCR Ingestion")
    uploaded = st.sidebar.file_uploader(
        "Upload a receipt image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if not uploaded:
        return

    st.sidebar.image(uploaded, caption="Uploaded receipt", use_container_width=True)
    if st.sidebar.button("Extract & Save", type="primary"):
        with st.spinner("Extracting receipt data with GPT-4o..."):
            client = get_client()
            extracted = extract_receipt_json_from_image(client, uploaded)
            extracted = coerce_extracted_fields(extracted)

            receipt_id = insert_receipt(
                conn,
                business_name=str(extracted.get("business_name") or ""),
                total=safe_float(extracted.get("total")),
                expense_date=str(extracted.get("expense_date") or ""),
                tin=str(extracted.get("tin") or ""),
                receipt_number=str(extracted.get("receipt_number") or ""),
                document_type=str(extracted.get("document_type") or "unknown"),
                description=str(extracted.get("description") or ""),
                raw_ocr_json=json.dumps(extracted, ensure_ascii=False),
            )

        st.sidebar.success(f"Saved to database (id={receipt_id}).")
        st.sidebar.json(extracted)
        st.rerun()


def finance_assistant_system_prompt(receipts: List[ReceiptRow]) -> str:
    table = receipts_to_markdown_table(receipts)
    return (
        "You are a Finance Assistant. You answer questions based on the provided receipt data.\n"
        "Reference specific business names and dates.\n"
        "Format currency as PHP X,XXX.XX.\n"
        "If asked about BIR compliance, highlight if TIN or receipt numbers are missing.\n\n"
        "Receipt data (most recent first):\n"
        f"{table}\n"
    )


def chat_reply(client: OpenAI, *, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    resp = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": system_prompt},
            *[
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("user", "assistant") and m.get("content")
            ],
        ],
    )
    return (resp.output_text or "").strip()


def main() -> None:
    st.set_page_config(page_title="Receipt AI Lite", layout="wide")
    st.title("Receipt AI Lite")
    st.caption("Upload receipts → extract fields → chat over your saved receipt database.")

    conn = get_conn()
    init_db(conn)

    render_uploader_sidebar(conn)
    render_recent_uploads_sidebar(conn)

    ensure_chat_state()

    receipts = fetch_receipts(conn)
    system_prompt = finance_assistant_system_prompt(receipts)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Ask about totals, categories, missing TIN/receipt #, etc.")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            try:
                client = get_client()
                answer = chat_reply(client, system_prompt=system_prompt, messages=st.session_state.messages)
                if not answer:
                    answer = "I couldn't generate a response. Please try again."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
