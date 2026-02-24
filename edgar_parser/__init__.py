"""edgar-parser: structured financial data from SEC EDGAR filings."""

from .pipeline import FilingNotFoundError, enrich_filing, parse_filing
from .matching import match_filing
