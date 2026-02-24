# edgar-parser

Structured financial data from SEC EDGAR filings.

Give your Python application access to income statements, balance sheets, cash flows, and qualitative sections from 10-Q, 10-K, and 8-K filings — parsed directly from SEC EDGAR with no third-party data vendor required.

This isn't a raw EDGAR scraper. The library handles XBRL namespace resolution, fiscal period matching, current vs. prior period alignment, sign normalization, and fuzzy metric lookup — so you get clean, analysis-ready data.

## What it does

**Core Extraction** (`parse_filing`)
- Parses iXBRL facts from 10-Q and 10-K filings on SEC EDGAR
- Resolves XBRL namespaces, units, scales, and dimensional qualifiers
- Enriches facts with presentation roles and negated-label sign flipping

**Period Matching** (`match_filing`)
- Aligns current and prior period values for each financial line item
- Handles quarterly, full-year, and 4Q (derived annual) modes
- Zip-matching with adaptive fallback to fuzzy matching for edge cases

**8-K Earnings Extraction** (`earnings_8k`) — *optional, requires `anthropic`*
- Extracts financial data from 8-K earnings press releases using Claude
- Automatic fallback when 10-Q/10-K is not yet available for a period
- Same output schema as core extraction for seamless integration

**Filing Sections** (`section_parser`)
- Parses qualitative sections from 10-K/10-Q filings (Risk Factors, MD&A, Business, etc.)
- Summary and full-text modes with configurable word limits

**High-Level Tools** (`tools`)
- `get_financials(ticker, year, quarter)` — all facts for a filing period
- `get_filings(ticker, year, quarter)` — list available filings (10-Q, 10-K, 8-K)
- `get_metric(ticker, year, quarter, metric_name)` — single metric lookup with fuzzy matching
- `get_filing_sections(ticker, year, quarter)` — qualitative section text

## Install

```bash
pip install edgar-parser
```

For 8-K earnings extraction (uses Claude API):

```bash
pip install "edgar-parser[llm]"
```

## Quick start

```python
from edgar_parser.tools import get_financials, get_metric

# Get all financial facts from Apple's Q1 2025 10-Q
result = get_financials("AAPL", 2025, 1)
for fact in result["facts"][:5]:
    print(f"{fact['metric']}: {fact['current_value']}")

# Look up a single metric
revenue = get_metric("AAPL", 2025, 1, "Revenue")
print(f"Revenue: {revenue['value']} {revenue['unit']}")
```

Or use the lower-level API for more control:

```python
from edgar_parser import parse_filing, match_filing

# Parse raw XBRL facts
parsed = parse_filing("AAPL", 2025, 1, full_year_mode=False)

# Match current vs. prior periods
matched = match_filing(parsed)
print(matched.head())
```

## Key functions

| Function | Module | Description |
|----------|--------|-------------|
| `get_financials` | `tools` | All facts for a ticker/period, with caching |
| `get_filings` | `tools` | List SEC filings (10-Q, 10-K, 8-K) for a period |
| `get_metric` | `tools` | Single metric lookup with fuzzy matching |
| `get_filing_sections` | `tools` | Qualitative section text (Risk Factors, MD&A, etc.) |
| `parse_filing` | `pipeline` | Low-level XBRL fact extraction |
| `enrich_filing` | `pipeline` | Fiscal period categorization and enrichment |
| `match_filing` | `matching` | Current vs. prior period alignment |
| `find_8k_for_period` | `earnings_8k` | Find 8-K earnings release for a fiscal period |

## Requirements

- Python 3.10+
- No API key needed — data comes directly from SEC EDGAR (public)
- Optional: `ANTHROPIC_API_KEY` environment variable for 8-K extraction
