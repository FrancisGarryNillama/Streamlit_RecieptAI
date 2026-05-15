import base64
from collections import Counter, defaultdict
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

try:
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None  # type: ignore

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


def parse_raw_ocr(raw_json: str) -> Dict[str, Any]:
    if not raw_json:
        return {}
    try:
        obj = json.loads(raw_json)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def receipt_to_finance_record(receipt: ReceiptRow) -> Dict[str, Any]:
    raw = parse_raw_ocr(receipt.raw_ocr_json)
    category = str(raw.get("expense_category") or "uncategorized").strip() or "uncategorized"
    if category not in EXPENSE_CATEGORIES:
        category = "uncategorized"

    date_value = None
    if receipt.expense_date:
        try:
            date_value = dt.date.fromisoformat(receipt.expense_date)
        except Exception:
            date_value = None

    merchant = receipt.business_name.strip() or "Unknown merchant"
    return {
        "id": receipt.id,
        "date": date_value,
        "date_label": receipt.expense_date or "Undated",
        "month": date_value.strftime("%Y-%m") if date_value else "Undated",
        "merchant": merchant,
        "category": category,
        "category_label": category.replace("_", " ").title(),
        "amount": float(receipt.total or 0.0),
        "tin": receipt.tin,
        "receipt_number": receipt.receipt_number,
        "document_type": receipt.document_type or "unknown",
        "description": receipt.description,
    }


def finance_records(receipts: List[ReceiptRow]) -> List[Dict[str, Any]]:
    return [receipt_to_finance_record(r) for r in receipts]


def records_dataframe(records: List[Dict[str, Any]]):
    if pd is None:
        return None
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def peso(value: float) -> str:
    return f"PHP {value:,.2f}"


def summarize_finances(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_spend = sum(float(r["amount"]) for r in records)
    categorized = Counter()
    merchant_counts = Counter()
    monthly = defaultdict(float)

    for record in records:
        amount = float(record["amount"])
        categorized[record["category_label"]] += amount
        merchant_counts[record["merchant"]] += 1
        if record["month"] != "Undated":
            monthly[record["month"]] += amount

    average_transaction = total_spend / len(records) if records else 0.0
    largest_category = categorized.most_common(1)[0] if categorized else ("No data", 0.0)
    recurring_merchants = [name for name, count in merchant_counts.items() if count >= 2]

    return {
        "total_spend": total_spend,
        "average_transaction": average_transaction,
        "largest_category": largest_category,
        "recurring_merchants": recurring_merchants[:5],
        "monthly": dict(sorted(monthly.items())),
        "category_totals": dict(categorized),
        "transaction_count": len(records),
    }


def detect_finance_alerts(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []
    amounts = [float(r["amount"]) for r in records if float(r["amount"]) > 0]
    average = sum(amounts) / len(amounts) if amounts else 0.0

    for record in records:
        amount = float(record["amount"])
        if average and amount >= average * 2.5 and amount >= 1000:
            alerts.append(
                {
                    "level": "High",
                    "title": "Large transaction detected",
                    "body": f"{record['merchant']} posted {peso(amount)}, well above your average transaction.",
                }
            )
        if not record["tin"] or not record["receipt_number"]:
            alerts.append(
                {
                    "level": "Review",
                    "title": "Missing receipt compliance field",
                    "body": f"{record['merchant']} is missing a TIN or receipt number.",
                }
            )

    summary = summarize_finances(records)
    if summary["recurring_merchants"]:
        alerts.append(
            {
                "level": "Info",
                "title": "Recurring merchants found",
                "body": "Possible subscriptions or repeat bills: "
                + ", ".join(summary["recurring_merchants"]),
            }
        )

    return alerts[:6]


def forecast_cash_flow(records: List[Dict[str, Any]], monthly_income: float) -> List[Dict[str, Any]]:
    summary = summarize_finances(records)
    monthly_values = list(summary["monthly"].values())
    avg_spend = sum(monthly_values) / len(monthly_values) if monthly_values else summary["total_spend"]
    avg_spend = avg_spend or 0.0

    today = dt.date.today()
    start_month = dt.date(today.year, today.month, 1)
    forecast = []
    balance = monthly_income - avg_spend
    for i in range(1, 7):
        month = start_month + dt.timedelta(days=32 * i)
        month = dt.date(month.year, month.month, 1)
        balance += monthly_income - avg_spend
        forecast.append(
            {
                "month": month.strftime("%Y-%m"),
                "projected_spend": round(avg_spend, 2),
                "projected_net_cash": round(balance, 2),
            }
        )
    return forecast


def render_metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brand_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --finance-green: #006400;
            --finance-gold: #FFD700;
            --finance-ink: #10251a;
            --finance-soft: #f7f9f4;
        }
        .stApp {
            background: linear-gradient(180deg, #fbfcf7 0%, #f2f7ed 100%);
            color: var(--finance-ink);
        }
        [data-testid="stSidebar"] {
            background: #eef5ea;
            border-right: 1px solid rgba(0, 100, 0, 0.16);
        }
        .finance-hero {
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(0, 100, 0, 0.18);
            border-radius: 8px;
            background: linear-gradient(135deg, #006400 0%, #0b7a2b 62%, #c5a900 100%);
            color: #ffffff;
            margin-bottom: 1rem;
        }
        .finance-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.2rem);
            letter-spacing: 0;
        }
        .finance-hero p {
            max-width: 760px;
            margin: 0.45rem 0 0;
            color: rgba(255, 255, 255, 0.86);
        }
        .metric-card, .alert-card, .module-panel {
            border: 1px solid rgba(0, 100, 0, 0.16);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 26px rgba(0, 64, 0, 0.06);
        }
        .metric-card {
            min-height: 126px;
            padding: 1rem;
            border-top: 4px solid var(--finance-gold);
        }
        .metric-label {
            color: #48624f;
            font-size: 0.82rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .metric-value {
            color: var(--finance-green);
            font-size: 1.65rem;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 0.45rem;
            overflow-wrap: anywhere;
        }
        .metric-caption {
            color: #6d7c70;
            font-size: 0.88rem;
            margin-top: 0.35rem;
        }
        .alert-card {
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            border-left: 5px solid var(--finance-gold);
        }
        .alert-level {
            color: var(--finance-green);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .alert-title {
            font-weight: 800;
            margin-top: 0.1rem;
        }
        .alert-body {
            color: #536358;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        .stButton button {
            border-radius: 8px;
            border-color: #006400;
        }
        .stProgress > div > div > div > div {
            background-color: var(--finance-gold);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_spending_breakdown(records: List[Dict[str, Any]]) -> None:
    st.subheader("Spending Breakdown")
    df = records_dataframe(records)
    if df is None or df.empty:
        st.info("Upload receipts to generate category and merchant analytics.")
        return

    category_df = (
        df.groupby("category_label", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    merchant_df = (
        df.groupby("merchant", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .head(8)
    )

    left, right = st.columns(2)
    with left:
        if px is not None:
            fig = px.pie(
                category_df,
                names="category_label",
                values="amount",
                hole=0.42,
                color_discrete_sequence=["#006400", "#FFD700", "#2e8b57", "#b59b00", "#77a86b"],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(category_df, use_container_width=True, hide_index=True)

    with right:
        if px is not None:
            fig = px.bar(
                merchant_df,
                x="amount",
                y="merchant",
                orientation="h",
                color_discrete_sequence=["#006400"],
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                height=360,
                xaxis_title="Spend",
                yaxis_title="Merchant",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(merchant_df.set_index("merchant"))


def render_savings_tracker(summary: Dict[str, Any], monthly_income: float, savings_goal: float) -> None:
    st.subheader("Savings Progress Tracker")
    estimated_savings = max(monthly_income - summary["total_spend"], 0.0)
    progress = min(estimated_savings / savings_goal, 1.0) if savings_goal else 0.0
    st.progress(progress, text=f"{peso(estimated_savings)} saved toward {peso(savings_goal)}")

    if estimated_savings <= 0 and monthly_income > 0:
        st.warning("Current receipt spend is above the monthly income target. Tighten discretionary categories first.")
    elif progress >= 1:
        st.success("Savings goal is on track based on current recorded spending.")
    else:
        gap = max(savings_goal - estimated_savings, 0.0)
        st.info(f"You need {peso(gap)} more to hit this savings goal.")


def render_cash_flow_forecast(records: List[Dict[str, Any]], monthly_income: float) -> None:
    st.subheader("Cash Flow Forecast")
    forecast = forecast_cash_flow(records, monthly_income)
    if pd is None:
        st.dataframe(forecast, use_container_width=True)
        return
    df = pd.DataFrame(forecast)
    if px is not None:
        fig = px.line(
            df,
            x="month",
            y=["projected_spend", "projected_net_cash"],
            markers=True,
            color_discrete_sequence=["#b59b00", "#006400"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            height=360,
            xaxis_title="Month",
            yaxis_title="PHP",
            legend_title_text="Forecast",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index("month"))


def render_alerts_panel(records: List[Dict[str, Any]]) -> None:
    st.subheader("Alerts & Notifications")
    alerts = detect_finance_alerts(records)
    if not alerts:
        st.success("No suspicious activity or compliance gaps detected yet.")
        return
    for alert in alerts:
        st.markdown(
            f"""
            <div class="alert-card">
                <div class="alert-level">{alert['level']}</div>
                <div class="alert-title">{alert['title']}</div>
                <div class="alert-body">{alert['body']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_investment_overview(monthly_income: float, summary: Dict[str, Any]) -> None:
    st.subheader("Investment Overview")
    investable = max(monthly_income - summary["total_spend"], 0.0)
    conservative = investable * 0.5
    balanced = investable * 0.3
    flexible = investable * 0.2

    data = [
        {"Allocation": "Emergency / Cash Buffer", "Amount": conservative},
        {"Allocation": "Index Funds / ETFs", "Amount": balanced},
        {"Allocation": "Flexible Goals", "Amount": flexible},
    ]
    if pd is not None:
        df = pd.DataFrame(data)
        if px is not None:
            fig = px.bar(
                df,
                x="Allocation",
                y="Amount",
                color="Allocation",
                color_discrete_sequence=["#006400", "#FFD700", "#2e8b57"],
            )
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df.set_index("Allocation"))
    st.caption("Educational allocation model only. It is not personalized investment, legal, or tax advice.")


def render_finance_dashboard(receipts: List[ReceiptRow]) -> None:
    records = finance_records(receipts)
    summary = summarize_finances(records)
    monthly_income = 50000.0
    savings_goal = 10000.0

    with st.sidebar:
        st.subheader("Finance Controls")
        st.markdown(f"**Monthly income target:** {peso(monthly_income)}")
        st.markdown(f"**Monthly savings goal:** {peso(savings_goal)}")
        st.caption("These planning assumptions keep the deployed dashboard stable and can be wired to account data later.")

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Total Spend", peso(summary["total_spend"]), f"{summary['transaction_count']} transactions")
    with cols[1]:
        render_metric_card("Avg Transaction", peso(summary["average_transaction"]), "Receipt-level average")
    with cols[2]:
        category, amount = summary["largest_category"]
        render_metric_card("Top Category", str(category), peso(float(amount)))
    with cols[3]:
        projected_save = max(monthly_income - summary["total_spend"], 0.0)
        render_metric_card("Projected Savings", peso(projected_save), "Income less recorded spend")

    st.divider()
    render_spending_breakdown(records)

    left, right = st.columns([1, 1])
    with left:
        render_savings_tracker(summary, monthly_income, savings_goal)
        render_alerts_panel(records)
    with right:
        render_cash_flow_forecast(records, monthly_income)
        render_investment_overview(monthly_income, summary)


def finance_assistant_system_prompt(receipts: List[ReceiptRow]) -> str:
    table = receipts_to_markdown_table(receipts)
    records = finance_records(receipts)
    summary = summarize_finances(records)
    alerts = detect_finance_alerts(records)
    return (
        "You are a Personal Finance Intelligence AI Assistant.\n"
        "Use the saved receipt and transaction data to provide budgeting, spending, fraud, cash-flow, savings, "
        "credit, lending, and investment education insights.\n"
        "Be practical, transparent, and proactive. Mention uncertainty when the data is incomplete.\n"
        "For lending and investment topics, explain tradeoffs clearly and avoid guaranteeing returns or approvals.\n"
        "Reference specific merchants and dates when useful.\n"
        "Format currency as PHP X,XXX.XX.\n"
        "If asked about automation, describe a recommended workflow rather than claiming you moved money.\n"
        "If asked about fraud, flag anomalies based on unusually large transactions, missing fields, or repeat merchants.\n\n"
        "Current analytics summary:\n"
        f"- Total spend: {peso(summary['total_spend'])}\n"
        f"- Average transaction: {peso(summary['average_transaction'])}\n"
        f"- Largest category: {summary['largest_category'][0]} ({peso(float(summary['largest_category'][1]))})\n"
        f"- Recurring merchants: {', '.join(summary['recurring_merchants']) or 'None detected'}\n"
        f"- Active alerts: {len(alerts)}\n\n"
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
    st.set_page_config(page_title="Personal Finance Intelligence AI", layout="wide")
    render_brand_styles()
    st.markdown(
        """
        <section class="finance-hero">
            <h1>Personal Finance Intelligence AI</h1>
            <p>
                Upload receipts, monitor spending, detect anomalies, forecast cash flow,
                and chat with an AI assistant about your financial decisions.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    conn = get_conn()
    init_db(conn)

    render_uploader_sidebar(conn)
    render_recent_uploads_sidebar(conn)

    ensure_chat_state()

    receipts = fetch_receipts(conn)
    system_prompt = finance_assistant_system_prompt(receipts)

    dashboard_tab, assistant_tab, transactions_tab = st.tabs(
        ["Finance Dashboard", "AI Assistant", "Transactions"]
    )

    with dashboard_tab:
        render_finance_dashboard(receipts)

    with assistant_tab:
        st.subheader("Conversational Finance Assistant")
        st.caption("Ask things like: Why did I spend more this month? How much can I save? Any suspicious activity?")

        starter_cols = st.columns(4)
        starter_prompts = [
            "Why did I spend more recently?",
            "How much can I save this month?",
            "Find suspicious transactions.",
            "What subscriptions or recurring bills do I have?",
        ]
        for idx, prompt in enumerate(starter_prompts):
            with starter_cols[idx]:
                if st.button(prompt, use_container_width=True):
                    st.session_state.pending_prompt = prompt

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_text = st.chat_input("Ask about spending, savings, fraud alerts, cash flow, credit, or investments.")
        if not user_text and st.session_state.get("pending_prompt"):
            user_text = st.session_state.pop("pending_prompt")

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

    with transactions_tab:
        st.subheader("Transaction Intelligence Table")
        records = finance_records(receipts)
        df = records_dataframe(records)
        if df is not None and not df.empty:
            visible_cols = [
                "date_label",
                "merchant",
                "category_label",
                "amount",
                "document_type",
                "receipt_number",
                "tin",
                "description",
            ]
            st.dataframe(df[visible_cols], use_container_width=True, hide_index=True)
        elif records:
            st.dataframe(records, use_container_width=True)
        else:
            st.info("No receipt transactions yet. Upload a JPG or PNG receipt from the sidebar.")


if __name__ == "__main__":
    main()
