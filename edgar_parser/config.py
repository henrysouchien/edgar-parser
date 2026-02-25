#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
import os
load_dotenv()

# === CONFIG & SETUP ==========================================

# === HEADERS ======
HEADERS = {
    "User-Agent": "edgar-parser henry@financialmodelupdater.com",
    "Accept-Encoding": "gzip, deflate",
}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

# === NUMBER OF FILINGS TO PULL ======
N_10Q = 12
N_10K = 4

# === EXTRA FILING PULLS ===
N_10Q_EXTRA = 0
N_10K_EXTRA = 0

# === SAFE LIMIT TIMES ===
REQUEST_DELAY = 1  # in seconds

# === HTTP CLIENT (timeouts/retries) ===
HTTP_CONNECT_TIMEOUT = float(os.getenv("EDGAR_HTTP_CONNECT_TIMEOUT", "5"))
HTTP_READ_TIMEOUT = float(os.getenv("EDGAR_HTTP_READ_TIMEOUT", "45"))
HTTP_MAX_RETRIES = int(os.getenv("EDGAR_HTTP_MAX_RETRIES", "2"))
HTTP_BACKOFF_FACTOR = float(os.getenv("EDGAR_HTTP_BACKOFF_FACTOR", "0.5"))
HTTP_ADAPTIVE_RATE_LIMIT = os.getenv("EDGAR_HTTP_ADAPTIVE_RATE_LIMIT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
HTTP_MIN_INTERVAL_SECONDS = float(
    os.getenv("EDGAR_HTTP_MIN_INTERVAL_SECONDS", str(REQUEST_DELAY))
)
HTTP_MIN_INTERVAL_DATA_SEC = float(
    os.getenv("EDGAR_HTTP_MIN_INTERVAL_DATA_SEC", str(HTTP_MIN_INTERVAL_SECONDS))
)
HTTP_MIN_INTERVAL_WWW_SEC = float(
    os.getenv("EDGAR_HTTP_MIN_INTERVAL_WWW_SEC", str(HTTP_MIN_INTERVAL_SECONDS))
)
ADAPTIVE_DISCOVERY = os.getenv("EDGAR_ADAPTIVE_DISCOVERY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
INFER_PERIOD_END_FROM_FILENAME = os.getenv("EDGAR_INFER_PERIOD_END_FROM_FILENAME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

# === EXPORTS ===
OUTPUT_METRICS_DIR = "metrics"
EXPORT_UPDATER_DIR = "exports"

# 8-K earnings release extraction
ANTHROPIC_MODEL_8K = "claude-sonnet-4-20250514"
MAX_8K_HTML_BYTES = 500_000

# OpenAI fallback for 8-K extraction
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_8K = os.getenv("EDGAR_OPENAI_MODEL_8K", "gpt-4o-2024-11-20")
ENABLE_8K_LLM_FALLBACK = os.getenv("EDGAR_8K_LLM_FALLBACK", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
MAX_8K_HTML_BYTES_OPENAI = int(os.getenv("EDGAR_MAX_8K_HTML_BYTES_OPENAI", "350000"))
