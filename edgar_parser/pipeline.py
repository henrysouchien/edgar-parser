"""Core extraction layer for EDGAR filing parsing."""

import gzip
import os
import re
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning

from .config import (
    ADAPTIVE_DISCOVERY,
    EXPORT_UPDATER_DIR,
    HEADERS,
    INFER_PERIOD_END_FROM_FILENAME,
    N_10K as N_10K_BASE,
    N_10K_EXTRA,
    N_10Q as N_10Q_BASE,
    N_10Q_EXTRA,
    OUTPUT_METRICS_DIR,
)
from .enrich import get_concept_roles_from_presentation, get_negated_label_concepts
from .http_client import get as http_get
from .utils import (
    AXIS_COLS,
    extract_dimensions_from_context,
    extract_fiscal_year_end,
    log_metric,
    lookup_cik_from_ticker,
    metrics,
    parse_date,
)

STOP_AFTER_FIRST_VALID_PERIOD = True


class FilingNotFoundError(ValueError):
    """Raised when requested 10-Q/10-K filing is not available."""


def enrich_filing(filing, results_10q, results_10k):

    """
    Enriches a parsed EDGAR 10-Q or 10-K filing by categorizing each XBRL fact into structured
    financial periods and tagging it with relevant metadata for modeling and analysis.

    This includes:
    - Resolving fiscal year and quarter start/end boundaries from prior filings
    - Mapping XBRL contextRefs to actual date ranges (duration or instant)
    - Assigning presentation roles and dimension axes (segment, geo, product, etc.)
    - Categorizing facts into matched time periods: current_q, prior_q, YTD, FY, etc.

    Args:
        filing (dict): A parsed filing dictionary containing:
            - 'facts': List of extracted XBRL facts
            - 'context_blocks': Dict of raw XBRL contextRef blocks
            - 'document_period_end': DEI DocumentPeriodEndDate (string)
            - 'form': Filing type (e.g., "10-Q" or "10-K")
            - 'accession': SEC accession number
            - 'concept_roles': Presentation role mapping from .pre.xml

    Returns:
        pandas.DataFrame: Enriched DataFrame of facts with additional fields:
            - tag, value, contextref
            - matched_category (e.g., current_q, prior_full_year, etc.)
            - date_type (e.g., Q, YTD, FY)
            - start, end, or instant date
            - presentation_role
            - axis_* fields: segment, product, geo, etc.
            - axis_unassigned: unclassified dimensional tags

    Notes:
        - Relies on external globals `results_10q` and `results_10k` for prior filing dates.
        - Includes fallbacks if certain filings or period matches are missing.
        - Automatically skips and warns on pre-2019 filings (inline XBRL not reliable).

    Example:
        df = enrich_filing(filing)
    """
    
    # === Step 1: Build reference dates from the filings ===
    
    # Get current period end date
    doc_end_date = parse_date(filing["document_period_end"])
    
    form = filing.get("form")  # "10-K" or "10-Q"
    prior_start_date = None
    prior_end_date = None

    # Sort filings by period end, descending
    sorted_10k = sorted(
        [f for f in results_10k if parse_date(f.get("document_period_end"))],
        key=lambda f: parse_date(f["document_period_end"]),
        reverse=True
    )
    
    sorted_10q = sorted(
        [f for f in results_10q if parse_date(f.get("document_period_end"))],
        key=lambda f: parse_date(f["document_period_end"]),
        reverse=True
    )
        
    # === Block to prevent extraction from filings before 2019 (no XBRL) ===
    if doc_end_date < parse_date("2019-01-01"):
        raise ValueError(
            f"⚡ Filing date {doc_end_date} is before 2019. "
            "This script only supports EDGAR filings from 2019 onward "
            "(inline XBRL not reliable before that). Please choose a filing from 2019 or later."
        )
    
    print("--------------------------------------------------")
    if filing.get("form") == "10-K":
        filing_label = f"FY{filing.get('year') % 100:02d}"
    else:
        filing_label = filing.get("label", "Unknown")
    
    print(f"\n🚀 Starting enrichment for {filing.get('form', 'Unknown')} [{filing_label}] | Period End: {filing.get('document_period_end', 'Unknown')} | Accession: {filing.get('accession', 'Unknown')}")
 

    # === Step 2: Get fiscal year start and end dates for current period and prior y/y period ===
    
    #Calculate fiscal year start with end date of prior 10K (prior fiscal year end date)
    
    # Sort 10-Ks by document_period_end descending
    sorted_10ks = sorted(results_10k, key=lambda x: parse_date(x["document_period_end"]), reverse=True)

    # Find prior 10-K end date (before doc_end_date)
    prior_10k_end_date = None
    for filing_prior in sorted_10ks:
        prior_end = parse_date(filing_prior["document_period_end"])
        if prior_end < doc_end_date:   
            prior_10k_end_date = prior_end  
            break

    # Fallback if not found
    if not prior_10k_end_date:
        print(f"⚠️ No prior 10-K found before {doc_end_date}. Using fallback prior 10-K end date with adjusted year.")
        prior_10k_end_date = doc_end_date.replace(year=doc_end_date.year - 1)

    # Use prior 10-K end to define fiscal year start
    prior_fiscal_year_end = prior_10k_end_date
    fiscal_year_start = prior_fiscal_year_end + timedelta(days=1)

    # Calculate prior fiscal year start with end date or 10-K before the prior 10-K (prior prior 10K end date)
    # To calculate the prior year start dates in filing
    
    prior_prior_10k_end_date = None
    for filing_prior2 in sorted_10ks:
        prior_end2 = parse_date(filing_prior2["document_period_end"])
        if prior_end2 and prior_end2 < prior_10k_end_date:
            prior_prior_10k_end_date = prior_end2
            break
    
    if not prior_prior_10k_end_date:
        print("⚠️ No second prior 10-K found — using fallback year subtraction.")
        prior_fiscal_year_start = fiscal_year_start.replace(year=fiscal_year_start.year - 1)
        
    else:
        prior_fiscal_year_start = prior_prior_10k_end_date + timedelta(days=1)

    # Set doc start date as fiscal year start if 10K filing
    if form == "10-K":
        doc_start_date = fiscal_year_start
        prior_start_date = prior_fiscal_year_start
        prior_end_date = prior_fiscal_year_end

    else:    
        
    # === Step 3: Get period start dates based on prior filings for 10Q's ===
        try:
            prior_filings = sorted(
                results_10q + results_10k,
                key=lambda x: parse_date(x.get("document_period_end")),
                reverse=True
            )
    
        except Exception as e:
            print(f"⚠️ Warning: Failed to sort filings by document_period_end. Error: {e}")
            prior_filings = []  # fallback to empty list
                
        doc_start_date = None
        for prior in prior_filings:
            try:
                candidate_end = parse_date(prior["document_period_end"])
                if candidate_end < doc_end_date:
                    doc_start_date = candidate_end + timedelta(days=1)
                    break
            except Exception:
                continue
            
        if not doc_start_date:
            doc_start_date = (doc_end_date - timedelta(days=90)).replace(day=1) #logic to use if no prior quarterly filings
            
    # === Step 4: Get prior y/y start and end dates from prior filings ===
    
    if form == "10-Q":
        quarter = filing.get("quarter")
        year = filing.get("year")

        prior_end_date = None
    
        if quarter and year:
            for q in results_10q:
                if (
                    q.get("quarter") == quarter
                    and q.get("year") == (year - 1)
                ):
                    q_end = parse_date(q["document_period_end"])
                    if q_end:
                        prior_end_date = q_end
                        break

        prior_start_date = None
        
        if prior_end_date:
            for prior in prior_filings:
                try:
                    candidate_end = parse_date(prior["document_period_end"])
                    if not candidate_end:
                        continue
                    if candidate_end < prior_end_date:
                        prior_start_date = candidate_end + timedelta(days=1)
                        break
                except Exception:
                    continue
        
    # Fallback: if either value is still missing
    if not prior_start_date or not prior_end_date:
        print("⚠️ prior_start_date or prior_end_date missing — applying YoY fallback")
        prior_start_date = doc_start_date.replace(year=doc_start_date.year - 1)
        prior_end_date = doc_end_date.replace(year=doc_end_date.year - 1)
    
    print(f"\n🎯 Current Period: {doc_start_date} to {doc_end_date} with fiscal year start {fiscal_year_start}")
    print(f"🎯 Prior Period:   {prior_start_date} to {prior_end_date} with fiscal year start {prior_fiscal_year_start}")
    
    # Step 5: Build context period lookup
    period_lookup = {}
    context_blocks = filing["context_blocks"]

    # === NEW: Extract dimension info per contextref ===
    context_dim_map = { 
        ctx_id: extract_dimensions_from_context(ctx_html)
        for ctx_id, ctx_html in context_blocks.items()
    }
    
    for ctx_id, block in context_blocks.items():
        if not ctx_id:
            continue
        if "<xbrli:startdate>" in block.lower() and "<xbrli:enddate>" in block.lower():
            start = re.search(r"<xbrli:startdate>(.*?)</xbrli:startdate>", block, re.IGNORECASE)
            end = re.search(r"<xbrli:enddate>(.*?)</xbrli:enddate>", block, re.IGNORECASE)
            if start and end:
                start = parse_date(start.group(1))
                end = parse_date(end.group(1))
                period_lookup[ctx_id] = ("duration", start, end)
        elif "<xbrli:instant>" in block.lower():
            instant = re.search(r"<xbrli:instant>(.*?)</xbrli:instant>", block, re.IGNORECASE)
            if instant:
                instant = parse_date(instant.group(1))
                period_lookup[ctx_id] = ("instant", instant)

    print(f"\n🧠 Mapped {len(period_lookup)} contextrefs to periods.")

    # Step 6: Enrich facts
    all_facts = []
    for fact in filing["facts"]:
        ctx = fact.get("contextref")
        tag = fact.get("tag")
        value = fact.get("value")
        
        if ctx not in period_lookup:
            continue
        
        period_info = period_lookup[ctx]
        enriched = {
            "tag": tag,
            "value": value,
            "contextref": ctx,
            "scale": fact.get("scale"),
            "period_type": period_info[0],
            "matched_category": None,
            "start": None,
            "end": None,
            "date_type": None,
            "presentation_role": None
        }

        # Assign presentation role if concept exists in pre.xml map
        roles = filing.get("concept_roles", {}).get(tag, [])
        enriched["presentation_role"] = (
            "|".join(sorted(set(r.lower() for r in roles if isinstance(r, str))))
            if roles else None
        )
        
        dims = context_dim_map.get(ctx, [])
        
        # Initialize axis category columns
        axis_columns = [
            "axis_consolidation",
            "axis_segment",
            "axis_product",
            "axis_geo",
            "axis_legal_entity",
            "axis_unassigned" 
        ]
        for col in axis_columns:
            enriched[col] = None
        
        # Smart dimension assignment (no mapping)
        for d in dims:
            axis = (d.get("dimension") or "").lower()
            member = d.get("member")
        
            if "consolidation" in axis:
                enriched["axis_consolidation"] = member
            elif "segment" in axis or "business" in axis:
                enriched["axis_segment"] = member
            elif "product" in axis or "service" in axis:
                enriched["axis_product"] = member
            elif "geo" in axis or "region" in axis or "country" in axis:
                enriched["axis_geo"] = member
            elif "legal" in axis or "entity" in axis:
                enriched["axis_legal_entity"] = member

        # === NEW: Catch-all for unclassified axes ===
        classified_keywords = ["consolidation", "segment", "business", "product", "service", "geo", "region", "country", "legal", "entity"]
        unclassified_dims = []
        
        for d in dims:
            axis = (d.get("dimension") or "").lower()
            member = d.get("member")
            if not any(k in axis for k in classified_keywords):
                unclassified_dims.append(f"{axis}={member}")
        
        enriched["axis_unassigned"] = "|".join(unclassified_dims) if unclassified_dims else None
        
        if period_info[0] == "duration": #Categorizing flow values (revenues, etc.) as current or prior FY, YTD, or Q periods
            start, end = period_info[1], period_info[2]
            enriched["start"] = start
            enriched["end"] = end

            if filing.get("form") == "10-K":
                if start == fiscal_year_start and end == doc_end_date:
                    enriched["matched_category"] = "current_full_year"
                elif start == prior_fiscal_year_start and end == prior_fiscal_year_end:
                    enriched["matched_category"] = "prior_full_year"

            else: # 10-Q logic
                if start == doc_start_date and end == doc_end_date:
                    enriched["matched_category"] = "current_q"
                elif start == fiscal_year_start and end == doc_end_date:
                    enriched["matched_category"] = "current_ytd"
                elif start == prior_start_date and end == prior_end_date:
                    enriched["matched_category"] = "prior_q"
                elif start == prior_fiscal_year_start and end == prior_end_date:
                    enriched["matched_category"] = "prior_ytd"
                
        elif period_info[0] == "instant": #Categorizing instant values (cash, etc.) as current or prior Q
            instant = period_info[1]
            enriched["end"] = instant
            if instant == doc_end_date:
                enriched["matched_category"] = "current_q"
            elif instant == prior_end_date:
                enriched["matched_category"] = "prior_q"

        # Categorize matched_category into simplified date_type
        mc = enriched["matched_category"]
        if mc in ["current_q", "prior_q"]:
            enriched["date_type"] = "Q"
        elif mc in ["current_ytd", "prior_ytd"]:
            enriched["date_type"] = "YTD"
        elif mc in ["current_full_year", "prior_full_year"]:
            enriched["date_type"] = "FY"
        else:
            enriched["date_type"] = None
        
        all_facts.append(enriched)

    print(f"\n✅ {len(all_facts)} facts extracted and enriched.")
    
    # Step 7: Build DataFrame
    df = pd.DataFrame(all_facts)
    print("\n🎯 Full categorization and enrichment complete!")
    print(f"✅ Completed enrichment for {filing.get('form', 'Unknown')} [{filing_label}] | Facts enriched: {len(all_facts)}")
    print("--------------------------------------------------")
    return df


def fetch_recent_10q_10k_accessions(cik, headers, n_10q, n_10k, min_10q=None, min_10k=None):

    """
    Fetches recent 10-Q and 10-K filings for a given company from the SEC EDGAR submissions JSON API.

    This function retrieves the company's real-time submission feed, filters for 10-Q and 10-K form types,
    and returns two lists of filing metadata, including accession numbers and report dates.

    Args:
        cik (str): The Central Index Key (CIK) of the company. Can be zero-padded or unpadded.
        headers (dict): HTTP headers for the request. Must include a valid 'User-Agent' per SEC guidelines.

    Returns:
        tuple of (list, list):
            - accessions_10q: List of dicts for each 10-Q filing, with keys:
                - 'accession': Accession number (e.g., "0001193125-23-123456")
                - 'report_date': Report period end date (YYYY-MM-DD)
                - 'form': Filing form ("10-Q")
            - accessions_10k: Same structure for 10-K filings.

    Notes:
        - The SEC submissions feed typically returns the most recent 250–1000 filings.
        - This function assumes standard JSON field structure from EDGAR and will raise an error if structure is missing.

    Example:
        accessions_10q, accessions_10k = fetch_recent_10q_10k_accessions("0000320193", headers)
    """
    
    def _extract_submission_arrays(payload, source):
        # Main submissions endpoint nests arrays under filings.recent.
        # Overflow archive files expose the same arrays at top level.
        filings = payload.get("filings", {}).get("recent")
        if filings is None:
            filings = payload

        required_keys = ["form", "accessionNumber", "reportDate"]
        if not all(k in filings for k in required_keys):
            raise ValueError(
                f"❌ SEC filings JSON missing expected fields (form, accessionNumber, reportDate) in {source}."
            )

        forms = filings["form"]
        accessions = filings["accessionNumber"]
        report_dates = filings["reportDate"]
        filing_dates = filings.get("filingDate", [None] * len(forms))
        return forms, accessions, report_dates, filing_dates

    def _scan_payload_for_10q_10k(payload, source, accessions_10q, accessions_10k, seen_accessions):
        forms, accessions, report_dates, filing_dates = _extract_submission_arrays(payload, source)

        for i, form in enumerate(forms):
            accession = accessions[i]
            if accession in seen_accessions:
                continue

            if form not in ("10-Q", "10-K"):
                continue

            seen_accessions.add(accession)
            entry = {
                "accession": accession,
                "report_date": report_dates[i],
                "filing_date": filing_dates[i],
                "form": form,
            }

            if form == "10-Q":
                accessions_10q.append(entry)
            else:
                accessions_10k.append(entry)

    def _overflow_file_url(cik_value, file_name):
        cik_10 = cik_value.zfill(10)
        if file_name.startswith("http://") or file_name.startswith("https://"):
            return file_name
        if file_name.startswith("CIK"):
            return f"https://data.sec.gov/submissions/{file_name}"
        if file_name.startswith("submissions-"):
            return f"https://data.sec.gov/submissions/CIK{cik_10}-{file_name}"
        return f"https://data.sec.gov/submissions/{file_name}"

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = http_get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    accessions_10q = []
    accessions_10k = []
    seen_accessions = set()

    _scan_payload_for_10q_10k(
        payload=data,
        source=f"CIK{cik_padded}.json",
        accessions_10q=accessions_10q,
        accessions_10k=accessions_10k,
        seen_accessions=seen_accessions,
    )

    target_min_10q = n_10q if min_10q is None else max(0, int(min_10q))
    target_min_10k = n_10k if min_10k is None else max(0, int(min_10k))

    # High-volume filers may have older submissions split into overflow files.
    if len(accessions_10q) < target_min_10q or len(accessions_10k) < target_min_10k:
        overflow_files = data.get("filings", {}).get("files", [])
        for file_meta in overflow_files:
            if len(accessions_10q) >= target_min_10q and len(accessions_10k) >= target_min_10k:
                break

            file_name = file_meta.get("name")
            if not file_name:
                continue

            overflow_url = _overflow_file_url(cik, file_name)
            overflow_resp = http_get(overflow_url, headers=headers, rate_limited=True)
            overflow_resp.raise_for_status()
            overflow_data = overflow_resp.json()

            _scan_payload_for_10q_10k(
                payload=overflow_data,
                source=file_name,
                accessions_10q=accessions_10q,
                accessions_10k=accessions_10k,
                seen_accessions=seen_accessions,
            )


    print(f"✅ Found {len(accessions_10q)} 10-Q accessions (filing submissions)")
    print(f"✅ Found {len(accessions_10k)} 10-K accessions (filing submissions)")
    
    return accessions_10q, accessions_10k


def filter_filings_by_year(accessions, max_year, n_limit):
    
    """
    Filters a list of filing accessions to include only those with a report_date in or before a specified year.
    Returns up to a maximum number of valid entries.

    Args:
        accessions (list of dict): List of filing entries, each containing a 'report_date' field in "YYYY-MM-DD" format.
        max_year (int): Latest acceptable year (inclusive).
        n_limit (int): Maximum number of entries to return after filtering.

    Returns:
        list of dict: Filtered list of filings matching the year condition, capped at n_limit.

    Notes:
        - Ignores filings with missing or invalid report dates.
        - Stops early once n_limit valid entries are found.

    Example:
        accessions_10q = filter_filings_by_year(accessions_10q, max_year=2023, n_limit=12)
    """
    
    filtered = []
    for entry in accessions:
        date_str = entry.get("report_date", "")
        if not date_str or date_str.strip() == "":
            continue
        try:
            yr = int(date_str.split("-")[0])
        except Exception:
            continue
        if yr > max_year:
            continue
        filtered.append(entry)
        if len(filtered) >= n_limit:
            break
    return filtered


def label_10q_accessions(accessions_10q, accessions_10k):
    
# === Labels 10-Q accessions from EDGAR with fiscal-end dates and assign quarters ===
    
    """
    Labels a list of 10-Q filings with fiscal year-end, quarter number, and standardized short labels (e.g. "2Q24")
    based on the company’s historical 10-K fiscal year-end dates.

    This function matches each 10-Q's report date to the closest fiscal year-end (from 10-Ks),
    calculates the time delta, and assigns:
        - 'fiscal_year_end' (datetime.date): Matched 10-K fiscal year end
        - 'quarter' (str): "Q1", "Q2", or "Q3"
        - 'label' (str): Shorthand like "1Q25" based on fiscal year
        - 'calendar_year' (int): Year of the 10-Q report date
        - Optional: 'non_standard_period' (True if delta does not match standard quarter logic)

    Args:
        accessions_10q (list of dict): 10-Q accessions, each with a 'report_date' (YYYY-MM-DD).
        accessions_10k (list of dict): 10-K accessions used to determine fiscal year boundaries.

    Returns:
        list of dict: Updated 10-Q accessions with new keys:
            'fiscal_year_end', 'quarter', 'label', 'calendar_year', and optionally 'non_standard_period'.

    Notes:
        - Fallback logic is used if a 10-Q date falls after the last known fiscal year-end.
        - Quarter mapping is based on the number of days between report date and matched fiscal year-end:
            • Q1 = ~270 days before FY end
            • Q2 = ~180 days before FY end
            • Q3 = ~90 days before FY end
        - This step is critical for pre-tagging and organizing 10-Qs before processing.

    Example:
        accessions_10q = label_10q_accessions(accessions_10q, accessions_10k)
        print(accessions_10q[0]["label"])  # → "2Q24"
    """

    # === Extract and sort valid fiscal year-end dates from 10-Ks ===
    fiscal_year_ends = []
    
    for entry in accessions_10k:
        fy_date = parse_date(entry["report_date"])
        if fy_date:
            fiscal_year_ends.append(fy_date)
    
    fiscal_year_ends = sorted(fiscal_year_ends, reverse=True)
    
    if not fiscal_year_ends:
        raise ValueError("No valid fiscal year-end dates found in 10-Ks.")
    
    # === Match each 10-Q to its fiscal year and assign quarter label ===
    print("\n📊 Matching 10-Qs to fiscal year-end and labeling quarters based off report date:")
    
    for q in accessions_10q:
        q_date = parse_date(q["report_date"])
        if not q_date:
            q["quarter"] = None
            q["label"] = None
            continue
    
        # Match to the fiscal year that this 10-Q falls into — first fiscal year-end after Q end
        
        # Prefer fiscal year-ends >= Q date (standard case)
        candidates = [fy for fy in fiscal_year_ends if fy >= q_date]
        
        if candidates:
            matched_fy = min(candidates)
            used_fallback = False
        else:
            # Fallback: use latest fiscal year-end before Q date
            candidates = [fy for fy in fiscal_year_ends if fy < q_date]
            matched_fy = max(candidates) if candidates else None
            used_fallback = True
        
        # 🧠 Shift forward if using fallback (e.g., using FY23 to label FY24)
        if matched_fy and used_fallback:
            matched_fy = matched_fy.replace(year=matched_fy.year + 1)
    
        if not matched_fy:
            print(f"⚠️ No matching fiscal year-end found for 10-Q ending {q['report_date']}")
            q["quarter"] = None
            q["label"] = None
            continue
    
        # Use day-based logic to assign correct quarter
        days_diff = (matched_fy - q_date).days
    
        if 70 <= days_diff <= 120:
            quarter = "Q3"
        elif 160 <= days_diff <= 200:
            quarter = "Q2"
        elif 250 <= days_diff <= 300:
            quarter = "Q1"
        else:
            print(f"⚠️ Unexpected delta ({days_diff} days) between {matched_fy.strftime('%Y-%m-%d')} and {q_date.strftime('%Y-%m-%d')} — nonstandard quarter")
            q["quarter"] = None
            q["label"] = None
            q["non_standard_period"] = True
            continue
    
        # Apply labels to the quarterly filings
        q["fiscal_year_end"] = matched_fy
        q["quarter"] = quarter
        q["calendar_year"] = q_date.year    #Note: Calendar year NOT fiscal year
        q["label"] = f"{quarter[1:]}Q{str(matched_fy.year)[-2:]}"  # e.g. "Q1" + "25" → "1Q25" #Uses fiscal year match
    
        print(f"✅ {q['report_date']} → {q['label']} (matched FY end {q['fiscal_year_end']})")
        
    return accessions_10q
    


def filter_10q_accessions(accessions_10q, fiscal_year, quarter):
    
    """
    Filters 10-Q filings to include only those relevant for the target fiscal year and quarter.

    This function constructs a list of (quarter, fiscal_year) pairs needed for quarterly workflows,
    including:
        - The target quarter (e.g., Q2 2025)
        - The prior quarter (e.g., Q1 2025)
        - The same quarter in the prior year (e.g., Q2 2024)
        - The prior quarter in the prior year (e.g., Q1 2024)

    For Q4 workflows, it includes Q3 and Q2 from both the current and prior fiscal years.

    Args:
        accessions_10q (list of dict): List of labeled 10-Q filings, each with 'quarter' and 'fiscal_year_end'.
        fiscal_year (int): Target fiscal year (e.g., 2025).
        quarter (int): Target fiscal quarter (1–4).

    Returns:
        list of dict: Filtered list of 10-Q accessions matching any of the (quarter, year) targets.

    Notes:
        - Only includes Q1–Q3 filings (Q4 is handled differently in most workflows).
        - Excludes filings with missing or malformed 'fiscal_year_end' or 'quarter'.
        - Relies on accurate labeling by `label_10q_accessions`.

    Example:
        required_10q = filter_10q_accessions(accessions_10q, fiscal_year=2025, quarter=2)
    """

    # === Build list of (quarter, fiscal_year) targets ===
    targets = []

    if quarter == 4:
        # Q3 and Q2 of current and prior fiscal years
        for q in [3, 2]:
            targets.append((f"Q{q}", fiscal_year))
            targets.append((f"Q{q}", fiscal_year - 1))
            
    else:
        # Target quarter
        targets.append((f"Q{quarter}", fiscal_year))
    
        # Prior quarter
        if quarter > 1:
            targets.append((f"Q{quarter - 1}", fiscal_year))
        else:
            targets.append(("Q4", fiscal_year - 1))
    
        # YoY same quarter
        targets.append((f"Q{quarter}", fiscal_year - 1))
    
        # YoY prior quarter
        if quarter > 1:
            targets.append((f"Q{quarter - 1}", fiscal_year - 1))
        else:
            targets.append(("Q4", fiscal_year - 2))

    # === Filter using parsed fiscal year from fiscal_year_end
    filtered = [
        q for q in accessions_10q
        if (
            q.get("quarter") in ["Q1", "Q2", "Q3"] and
            q.get("fiscal_year_end") and
            (q["quarter"], q["fiscal_year_end"].year) in targets
        )
    ]

    print(f"✅ Selected {len(filtered)} 10-Q filings for processing.")
    return filtered
    


def enrich_10k_accessions_with_fiscal_year(accessions_10k):
# === Enrich 10-K results with fiscal year metadata ===
# Gathers fiscal year end date from recent 10-K filings
    
    """
    Enriches each 10-K filing in the list with fiscal year metadata based on its report date.

    For each 10-K filing, this function parses the 'report_date' (or 'document_period_end' if available)
    and assigns:
        - 'year': Fiscal year (YYYY), inferred directly from the date
        - 'fiscal_year_end': Full fiscal year-end date as a datetime.date object

    This enrichment step allows for proper fiscal alignment and quarter labeling of related filings
    without requiring .htm downloads or full XBRL parsing.

    Args:
        accessions_10k (list of dict): List of 10-K filings, each containing at least:
            - 'report_date' (str): Period end date in YYYY-MM-DD format.

    Returns:
        list of dict: Same list with added keys:
            - 'year': Fiscal year as integer
            - 'fiscal_year_end': datetime.date object for year-end

    Notes:
        - If parsing fails, 'year' and 'fiscal_year_end' are set to None.
        - Used upstream for quarter alignment and fiscal boundary inference.

    Example:
        enriched_10k = enrich_10k_accessions_with_fiscal_year(accessions_10k)
        print(enriched_10k[0]["year"])  # → 2023
        print(enriched_10k[0]["fiscal_year_end"])  # → datetime.date(2023, 12, 31)
    """

    print("\n🛠 Enriching 10-Ks with fiscal year and fiscal year-end...")
    
    for k in accessions_10k:
        period_end = k.get("report_date")
        dt = parse_date(period_end)
    
        if dt:
            k["year"] = dt.year # note this is FISCAL year
            k["fiscal_year_end"] = dt
            print(f"✅ {period_end} → Fiscal Year {k['year']} | Fiscal Year End: {k['fiscal_year_end']}")
        else:
            k["year"] = None
            k["fiscal_year_end"] = None
            print(f"⚠️ Could not parse period end for accession {k['accession']}")
            
    return accessions_10k



def filter_10k_accessions(accessions_10k, fiscal_year, quarter):
    
    """
    Filters a list of 10-K filings to include only those needed for the target fiscal year and quarter.

    For Q4 workflows, includes 10-Ks for:
        - the current fiscal year
        - the prior fiscal year
        - the prior-prior fiscal year

    For Q1–Q3 workflows, includes 10-Ks only for:
        - the prior fiscal year
        - the prior-prior fiscal year

    This ensures sufficient historical context for full-year rollforward, YoY comparisons, and fiscal 
    year boundary resolution without pulling unnecessary filings.

    Args:
        accessions_10k (list of dict): 10-K filings with a 'year' field assigned via enrichment.
        fiscal_year (int): Target fiscal year (e.g., 2025).
        quarter (int): Target fiscal quarter (1–4).

    Returns:
        list of dict: Filtered 10-K filings relevant to the current workflow.

    Example:
        required_10k = filter_10k_accessions(accessions_10k, fiscal_year=2025, quarter=4)
    """

    if quarter == 4:
        needed_years = {fiscal_year, fiscal_year - 1, fiscal_year - 2}
    else:
        needed_years = {fiscal_year - 1, fiscal_year - 2} #Required 10-K's for quarterly workflow

    filtered = [
        k for k in accessions_10k
        if (
            k["year"] in needed_years
        )
    ]

    print(f"✅ Selected {len(filtered)} 10-K filings for processing.")
    return filtered


def _enrich_parsed_10k_results(results_10k):
    """Attach fiscal year metadata to parsed 10-K results in-place."""
    for filing in results_10k:
        period_end = filing.get("document_period_end")
        dt = parse_date(period_end)
        if dt:
            filing["year"] = dt.year
            filing["fiscal_year_end"] = dt
        else:
            filing["year"] = None
            filing["fiscal_year_end"] = None
    return results_10k


def _label_parsed_10q_results(results_10q, results_10k):
    """Label parsed 10-Q results using parsed 10-K fiscal year ends."""
    fiscal_year_ends = []
    for entry in results_10k:
        fy_date = parse_date(entry.get("document_period_end"))
        if fy_date:
            fiscal_year_ends.append(fy_date)
    fiscal_year_ends = sorted(fiscal_year_ends, reverse=True)
    if not fiscal_year_ends:
        return False

    for filing in results_10q:
        q_date = parse_date(filing.get("document_period_end"))
        if not q_date:
            filing["quarter"] = None
            filing["label"] = None
            continue

        candidates = [fy for fy in fiscal_year_ends if fy >= q_date]
        if candidates:
            matched_fy = min(candidates)
            used_fallback = False
        else:
            candidates = [fy for fy in fiscal_year_ends if fy < q_date]
            matched_fy = max(candidates) if candidates else None
            used_fallback = True

        if matched_fy and used_fallback:
            matched_fy = matched_fy.replace(year=matched_fy.year + 1)

        if not matched_fy:
            filing["quarter"] = None
            filing["label"] = None
            continue

        days_diff = (matched_fy - q_date).days
        if 70 <= days_diff <= 120:
            quarter_label = "Q3"
        elif 160 <= days_diff <= 200:
            quarter_label = "Q2"
        elif 250 <= days_diff <= 300:
            quarter_label = "Q1"
        else:
            filing["quarter"] = None
            filing["label"] = None
            filing["non_standard_period"] = True
            continue

        filing["fiscal_year_end"] = matched_fy
        filing["quarter"] = quarter_label
        filing["year"] = q_date.year
        filing["label"] = f"{quarter_label[1:]}Q{str(matched_fy.year)[-2:]}"

    return True


def _fallback_coverage_satisfied(results_10q, results_10k, fiscal_year, quarter, four_q_mode):
    """Check whether parsed fallback results satisfy downstream filing dependencies."""
    if not results_10k:
        return False, "no parsed 10-K results"

    _enrich_parsed_10k_results(results_10k)
    if not _label_parsed_10q_results(results_10q, results_10k):
        return False, "unable to label 10-Q results from parsed 10-K anchors"

    if four_q_mode:
        target_10k = next((k for k in results_10k if k.get("year") == fiscal_year), None)
        prior_10k = next((k for k in results_10k if k.get("year") == (fiscal_year - 1)), None)
        if not target_10k:
            return False, f"missing target 10-K year {fiscal_year}"
        if not prior_10k:
            return False, f"missing prior 10-K year {fiscal_year - 1}"

        fye_target = target_10k.get("fiscal_year_end")
        fye_prior = prior_10k.get("fiscal_year_end")
        q3_current = next(
            (q for q in results_10q if q.get("quarter") == "Q3" and q.get("fiscal_year_end") == fye_target),
            None,
        )
        q3_prior = next(
            (q for q in results_10q if q.get("quarter") == "Q3" and q.get("fiscal_year_end") == fye_prior),
            None,
        )
        if not q3_current:
            return False, f"missing current-year Q3 for FY {fiscal_year}"
        if not q3_prior:
            return False, f"missing prior-year Q3 for FY {fiscal_year - 1}"
        return True, "4Q coverage satisfied"

    target_label = f"{quarter}Q{str(fiscal_year)[-2:]}"
    target_10q = next((q for q in results_10q if q.get("label") == target_label), None)
    if not target_10q:
        return False, f"missing target 10-Q label {target_label}"

    target_quarter = target_10q.get("quarter")
    target_fye = target_10q.get("fiscal_year_end")
    prior_10q = next(
        (
            q for q in results_10q
            if q.get("quarter") == target_quarter
            and q.get("fiscal_year_end")
            and target_fye
            and q["fiscal_year_end"].year == (target_fye.year - 1)
        ),
        None,
    )
    if not prior_10q:
        return False, f"missing prior-year 10-Q for quarter {target_quarter}"

    return True, "quarterly coverage satisfied"


def _remaining_entries(full_entries, selected_entries):
    """Return entries not included in the selected subset (by accession)."""
    selected_accessions = {entry.get("accession") for entry in selected_entries if entry.get("accession")}
    return [entry for entry in full_entries if entry.get("accession") not in selected_accessions]


def _merge_entries_by_accession(primary_entries, fallback_entries, max_items=None):
    """Merge entry lists while preserving order and de-duplicating by accession."""
    merged = []
    seen_accessions = set()
    for entry in list(primary_entries) + list(fallback_entries):
        accession = entry.get("accession")
        if accession:
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)
        merged.append(entry)
        if max_items is not None and len(merged) >= max_items:
            break
    return merged


def _sort_results_by_accession_order(results, source_entries):
    """Sort parsed results by accession order from the original source list."""
    accession_order = {
        entry.get("accession"): idx for idx, entry in enumerate(source_entries) if entry.get("accession")
    }
    return sorted(results, key=lambda row: accession_order.get(row.get("accession"), len(accession_order)))


def _sort_accessions_by_report_date_desc(accessions):
    """Sort accession metadata newest-first using report_date."""
    return sorted(accessions, key=lambda entry: entry.get("report_date", ""), reverse=True)


def fetch_10q_10k_accessions_from_master (cik, headers, years=None, quarters=None):
    
    """
    Retrieves 10-Q and 10-K filings for a given company by scanning the SEC EDGAR master index archive.

    This function downloads and parses master index files (`master.gz`) across the specified years and
    quarters, filters for filings matching the provided CIK and form types ("10-Q", "10-K"), and returns
    structured metadata for each matching filing.

    Args:
        cik (str): SEC Central Index Key for the company. Leading zeros are optional.
        headers (dict): HTTP headers, must include 'User-Agent' as required by the SEC.
        years (list of int): List of years to query (e.g., [2022, 2023]).
        quarters (list of str): List of calendar quarters to query (e.g., ["QTR1", "QTR2"]).

    Returns:
        tuple of (list, list):
            - accessions_10q: List of 10-Q filings, each with:
                - 'accession': Accession number (e.g., "0000320193-23-000055")
                - 'report_date': Filing date (used as report proxy)
                - 'form': Filing type ("10-Q")
            - accessions_10k: Same structure for 10-K filings.

    Notes:
        - Uses the filing date as 'report_date' to remain compatible with downstream workflows.
        - Skips any malformed lines or non-matching forms/CIKs silently.
        - May return a large number of filings if scanning across many years/quarters.

    Example:
        accessions_10q, accessions_10k = fetch_10q_10k_accessions_from_master(
            cik="0000320193",
            headers=headers,
            years=[2023],
            quarters=["QTR1", "QTR2"]
        )
    """
    
    cik_str = str(cik).lstrip("0") # normalize
    
    accessions_10q = []
    accessions_10k = []

    for year in years:
        for qtr in quarters:
            url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/{qtr}/master.gz"
            print(f"📦 Downloading: {url}")
            try:
                r = http_get(url, headers=headers, rate_limited=True)
                r.raise_for_status()
            except Exception as e:
                print(f"❌ Failed to fetch {year} {qtr}: {e}")
                continue

            # Decompress and decode
            with gzip.open(BytesIO(r.content), 'rt', encoding='latin-1') as f:
                started = False
                for line in f:
                    if not started:
                        if line.strip().startswith("CIK|"):
                            started = True
                        continue

                    parts = line.strip().split("|")
                    if len(parts) != 5:
                        continue

                    cik_field, company, form, date_filed, filename = parts

                    if cik_field != cik_str:
                        continue  # skip other companies

                    if form not in ("10-Q", "10-K"):
                        continue  # skip other forms

                    accession = filename.split("/")[-1].replace(".txt", "")
                    entry = {
                        "accession": accession,
                        "report_date": date_filed, #this is the filing date - but using report_date preserve logic downstream
                        "form": form
                    }

                    if form == "10-Q":
                        accessions_10q.append(entry)
                    elif form == "10-K":
                        accessions_10k.append(entry)

    accessions_10q = _sort_accessions_by_report_date_desc(accessions_10q)
    accessions_10k = _sort_accessions_by_report_date_desc(accessions_10k)
    print(f"✅ Found {len(accessions_10q)} 10-Q accessions (from master index)")
    print(f"✅ Found {len(accessions_10k)} 10-K accessions (from master index)")
    return accessions_10q, accessions_10k


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _compute_percentile(values, percentile):
    """Return percentile from a numeric list using linear interpolation."""
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if upper_index == lower_index:
        return float(lower_value)
    weight = rank - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def _infer_document_period_end_from_filename(filename, report_date=None):
    """Infer YYYY-MM-DD period end from filename date tokens when confidence is high."""
    name = str(filename or "").lower()
    date_tokens = re.findall(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", name)
    if not date_tokens:
        return None

    parsed_report_date = parse_date(report_date) if report_date else None
    candidates = []
    for year_str, month_str, day_str in date_tokens:
        try:
            inferred_dt = datetime(int(year_str), int(month_str), int(day_str)).date()
        except ValueError:
            continue
        if parsed_report_date:
            if inferred_dt > parsed_report_date:
                continue
            if (parsed_report_date - inferred_dt).days > 400:
                continue
        candidates.append(inferred_dt)

    if not candidates:
        return None
    return min(candidates).isoformat()



def extract_facts_with_document_period(ixbrl_url, headers):

    """
    Downloads and parses a single inline XBRL (.htm) filing from EDGAR to extract structured
    financial facts, context metadata, and the document period end date (DocumentPeriodEndDate).

    This function performs the following:
        - Fetches and parses the iXBRL content from the provided SEC URL
        - Extracts all <ix:nonfraction> and <ix:nonnumeric> tagged facts
        - Identifies and stores all <xbrli:context> blocks for future dimensional mapping
        - Captures and returns the DocumentPeriodEndDate (DEI tag) as the fiscal anchor

    Args:
        ixbrl_url (str): Full URL to the iXBRL .htm filing (e.g., from SEC index.json).
        headers (dict): HTTP headers to include in the request (must include 'User-Agent').

    Returns:
        dict: A dictionary containing:
            - 'facts': List of fact dictionaries with 'tag', 'contextref', 'value', and 'text'
            - 'context_blocks': Dict of raw <xbrli:context> blocks keyed by contextRef ID
            - 'document_period_end': DEI DocumentPeriodEndDate as a string (e.g., "2023-12-31")
            - 'document_period_label': Human-readable version of the period end (if available)

    Notes:
        - Applies sign reversal for facts tagged with `sign="-"` as per iXBRL convention
        - Skips facts with missing tag name, contextref, or value
        - Skips malformed numeric values that cannot be parsed
        - Warns on unusually large filings or fact count (>800) for performance awareness

    Example:
        data = extract_facts_with_document_period("https://www.sec.gov/Archives/...", headers)
        print(data["document_period_end"])  # → "2023-06-30"
    """
    
    print(f"\n🌐 Fetching iXBRL: {ixbrl_url}")
    t0 = time.time()

    r = http_get(ixbrl_url, headers=headers, rate_limited=True)
    fetch_time = time.time() - t0
    print(f"⏳ Fetch time: {fetch_time:.2f} seconds")

    r.raise_for_status()

    t1 = time.time()
    soup = BeautifulSoup(r.content, "lxml")
    parse_time = time.time() - t1
    print(f"🧠 Parse time: {parse_time:.2f} seconds")

    # === Dynamic slowdown warning ===
    content_mb = len(r.content) / 1_000_000
    if content_mb > 3:
        print(f"⚠️ Large filing ({content_mb:.1f} MB) — this may take a minute...")

    ix_tags = soup.find_all(["ix:nonfraction", "ix:nonnumeric"])
    print(f"📦 Found {len(ix_tags)} ix: tags")
    
    if len(ix_tags) > 800:
        print(f"⚠️ Detected {len(ix_tags)} facts — parsing may take a minute...")
    
    facts = []
    context_blocks = {}  # 🆕 New dictionary to store contexts
    doc_period_end = None
    doc_period_label = None

    # --- First: Extract all context blocks ---
    for ctx_tag in soup.find_all("xbrli:context"):
        ctx_id = ctx_tag.get("id")
        if ctx_id:
            context_blocks[ctx_id] = str(ctx_tag)  # Save the raw HTML block

    # --- Then: Extract all facts ---
    for tag in ix_tags:
        name = tag.get("name")
        ctx = tag.get("contextref")
        sign = tag.get("sign")
        scale = tag.get("scale")
        val = tag.text or tag.get("value") or "".join(tag.stripped_strings)

        if not (name and ctx and val):
            continue

        if name == "dei:DocumentPeriodEndDate":
            doc_period_end = val.strip()
            doc_period_label = tag.text.strip()

        try:
            value = float(val.replace(",", "").replace("−", "-"))
            if sign == "-":  # ✅ New: apply sign flip
                value = -abs(value)
        except ValueError:
            continue

        facts.append({
            "tag": name,
            "contextref": ctx,
            "value": value,
            "text": tag.text.strip(),
            "scale": _safe_int(scale)
        })

    return {
        "facts": facts,
        "context_blocks": context_blocks,  # 🆕 Include context_blocks in the return!
        "document_period_end": doc_period_end,
        "document_period_label": doc_period_label
    }


def try_all_htm_files(cik, accession_number, headers, profile=None, report_date=None):

    """
    Attempts to extract structured financial data from all .htm files within a specific SEC EDGAR filing accession.

    This function prioritizes the largest .htm file (by byte size) under the assumption it is most likely to
    contain the full iXBRL filing. If that file is invalid or incomplete, it falls back to scanning all other
    .htm files in the accession folder until it finds one with a valid DocumentPeriodEndDate and ≥50 extracted facts.

    Args:
        cik (str or int): Central Index Key (CIK) for the company.
        accession_number (str): SEC accession number (e.g., "0000320193-23-000055").
        headers (dict): HTTP headers for SEC API requests. Must include 'User-Agent'.

    Returns:
        list of dict: A list (typically with one entry) of extracted data from a valid .htm file, each containing:
            - 'file': Filename of the parsed .htm file
            - 'url': Full SEC URL of the .htm
            - 'document_period_end': DEI tag (e.g. "2023-06-30")
            - 'document_period_label': Human-readable date label (if present)
            - 'facts': List of extracted financial facts (tag, contextref, value, text)
            - 'context_blocks': Raw XBRL context blocks used for dimensional labeling
            - 'concept_roles': Mapping of tags to their presentation roles (from .pre.xml)

    Behavior:
        - Automatically skips .htm files with <50 extracted facts (likely exhibits or junk files).
        - Stops at the first valid .htm file unless STOP_AFTER_FIRST_VALID_PERIOD is False.
        - Applies fallback logic if the largest file is invalid, attempting all remaining .htms.

    Example:
        results = try_all_htm_files("320193", "0000320193-23-000055", headers)
        print(results[0]["document_period_end"])  # → "2023-12-31"
    """

    profile_data = {
        "accession": accession_number,
        "index_fetch_seconds": 0.0,
        "index_fetch_ok": False,
        "htm_candidates": 0,
        "htm_attempts": 0,
        "htm_attempt_seconds_total": 0.0,
        "used_fallback_scan": False,
        "selected_file": None,
        "selected_fact_count": 0,
        "inferred_period_end_used": False,
    }

    acc_nodash = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/index.json"
    base_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"

    index_start = time.time()
    try:
        r = http_get(index_url, headers=headers, rate_limited=True)
        r.raise_for_status()
        index = r.json()
        profile_data["index_fetch_ok"] = True
    except Exception as e:
        print(f"❌ Failed to fetch index.json for {accession_number}: {e}")
        profile_data["index_fetch_seconds"] = round(time.time() - index_start, 4)
        if profile is not None:
            profile.update(profile_data)
        return []
    profile_data["index_fetch_seconds"] = round(time.time() - index_start, 4)

    items = index.get("directory", {}).get("item", [])
    results = []

    # === Try largest .htm file by size first ===
    htm_items = [
        item for item in items
        if item["name"].lower().endswith(".htm") and item.get("size", "").isdigit()
    ]
    profile_data["htm_candidates"] = len(htm_items)

    tried_htm_names = set()
    if htm_items:
        # Sort descending by file size
        htm_items.sort(key=lambda x: int(x["size"]), reverse=True)
        largest_htm = htm_items[0]["name"]
        tried_htm_names.add(largest_htm)
        full_url = base_url + largest_htm
        print(f"📏 Trying largest .htm file first: {largest_htm} ({htm_items[0]['size']} bytes)")

        try:
            attempt_start = time.time()
            data = extract_facts_with_document_period(full_url, headers)
            profile_data["htm_attempts"] += 1
            profile_data["htm_attempt_seconds_total"] += time.time() - attempt_start

            period_end = data.get("document_period_end")
            if (
                not period_end
                and INFER_PERIOD_END_FROM_FILENAME
                and len(data.get("facts", [])) >= 1000
            ):
                inferred_period_end = _infer_document_period_end_from_filename(
                    largest_htm,
                    report_date=report_date,
                )
                if inferred_period_end:
                    period_end = inferred_period_end
                    data["document_period_end"] = inferred_period_end
                    data["document_period_label"] = inferred_period_end
                    profile_data["inferred_period_end_used"] = True

            if period_end and len(data["facts"]) >= 50:
                # Fetch presentation roles from .pre.xml
                concept_roles = get_concept_roles_from_presentation(cik, accession_number, headers)

                print(f"✅ {full_url} → Period End: {data['document_period_end']}")
                print(f"🔎 Extracted {len(data['facts'])} facts")

                profile_data["selected_file"] = largest_htm
                profile_data["selected_fact_count"] = len(data["facts"])
                results.append({
                    "file": largest_htm,
                    "url": full_url,
                    "document_period_end": data["document_period_end"],
                    "document_period_label": data["document_period_label"],
                    "facts": data["facts"],
                    "context_blocks": data["context_blocks"],
                    "concept_roles": concept_roles,
                })
                profile_data["htm_attempt_seconds_total"] = round(profile_data["htm_attempt_seconds_total"], 4)
                if profile is not None:
                    profile.update(profile_data)
                return results  # ✅ Success: stop here

        except Exception as e:
            print(f"⚠️ Error checking largest .htm file: {e}")

    # === Fallback: Try all other .htm files ===
    print("🔁 Fallback: checking all .htm files...")
    profile_data["used_fallback_scan"] = True
    fallback_htm_items = [
        item for item in items
        if item.get("name", "").lower().endswith(".htm") and item.get("name") not in tried_htm_names
    ]
    fallback_htm_items.sort(
        key=lambda item: int(item["size"]) if str(item.get("size", "")).isdigit() else -1,
        reverse=True,
    )
    for item in fallback_htm_items:

        full_url = base_url + item["name"]
        try:
            attempt_start = time.time()
            data = extract_facts_with_document_period(full_url, headers)
            profile_data["htm_attempts"] += 1
            profile_data["htm_attempt_seconds_total"] += time.time() - attempt_start

            period_end = data.get("document_period_end")
            if (
                not period_end
                and INFER_PERIOD_END_FROM_FILENAME
                and len(data.get("facts", [])) >= 1000
            ):
                inferred_period_end = _infer_document_period_end_from_filename(
                    item["name"],
                    report_date=report_date,
                )
                if inferred_period_end:
                    period_end = inferred_period_end
                    data["document_period_end"] = inferred_period_end
                    data["document_period_label"] = inferred_period_end
                    profile_data["inferred_period_end_used"] = True

            if period_end:
                if len(data["facts"]) < 50:
                    print(
                        f"⚠️ Warning: only {len(data['facts'])} facts extracted from {full_url} — possible exhibit or junk file."
                    )
                    continue  # Skip this .htm and keep looking

                # Fetch presentation roles from .pre.xml
                concept_roles = get_concept_roles_from_presentation(cik, accession_number, headers)

                print(f"✅ {full_url} → Period End: {data['document_period_end']}")
                print(f"🔎 Extracted {len(data['facts'])} facts")

                profile_data["selected_file"] = item["name"]
                profile_data["selected_fact_count"] = len(data["facts"])
                results.append({
                    "file": item["name"],
                    "url": full_url,
                    "document_period_end": data["document_period_end"],
                    "document_period_label": data["document_period_label"],
                    "facts": data["facts"],
                    "context_blocks": data["context_blocks"],  # 🆕 Capture context blocks too
                    "concept_roles": concept_roles,
                })
                if STOP_AFTER_FIRST_VALID_PERIOD:
                    break  # 🔥 Stop scanning more .htms once one good file is found

        except Exception as e:
            print(f"⚠️ Error checking {item['name']}: {e}")
            continue

    profile_data["htm_attempt_seconds_total"] = round(profile_data["htm_attempt_seconds_total"], 4)
    if profile is not None:
        profile.update(profile_data)
    return results


def extract_filing_batch(accessions, cik, headers, form_type):

    """
    Extracts financial data from a batch of EDGAR 10-Q or 10-K filings using inline XBRL (.htm) files.

    For each filing in the accession list:
      - Skips filings before 2019 (inline XBRL not guaranteed)
      - Calls `try_all_htm_files()` to locate and parse the first valid .htm file
      - Aggregates extracted data including facts, context blocks, and presentation roles

    Args:
        accessions (list of dict): Filing metadata including 'accession' and 'report_date'.
        cik (str or int): SEC Central Index Key for the company.
        headers (dict): HTTP headers for SEC requests (must include 'User-Agent').
        form_type (str): Filing type label, e.g. "10-Q" or "10-K".

    Returns:
        list of dict: One entry per successfully parsed filing, each containing:
            - 'accession': Accession number (e.g., "0001193125-23-123456")
            - 'report_date': Filing's report or submission date
            - 'file': Name of the parsed .htm file
            - 'url': Full SEC URL to the .htm file
            - 'document_period_end': DEI period end date (as string)
            - 'document_period_label': Human-readable label for the filing period
            - 'facts': List of extracted financial fact dicts
            - 'context_blocks': Raw XBRL context XML blocks
            - 'concept_roles': Presentation roles from .pre.xml
            - 'form': Filing type ("10-Q" or "10-K")

    Notes:
        - Uses `try_all_htm_files()` for efficient, prioritized .htm scanning.
        - Returns only filings that contain ≥50 iXBRL facts and a valid period end.

    Example:
        results_10q = extract_filing_batch(required_10q_filings, "320193", headers, "10-Q")
    """
    
    results = []
    profile_rows = []
    for i, entry in enumerate(accessions):
        acc = entry["accession"]
        report_date = entry["report_date"]
        accession_start = time.time()
        profile = {
            "accession": acc,
            "report_date": report_date,
            "pre_2019_skipped": False,
            "success": False,
            "result_count": 0,
            "elapsed_seconds": 0.0,
            "index_fetch_seconds": 0.0,
            "index_fetch_ok": False,
            "htm_candidates": 0,
            "htm_attempts": 0,
            "htm_attempt_seconds_total": 0.0,
            "used_fallback_scan": False,
            "selected_file": None,
            "selected_fact_count": 0,
            "inferred_period_end_used": False,
        }

        # 🚫 Skip filings before 2019 (no inline XBRL guaranteed)
        if int(report_date[:4]) < 2019:
            print(f"⏩ Skipping {acc} — pre-2019 filing")
            profile["pre_2019_skipped"] = True
            profile["elapsed_seconds"] = round(time.time() - accession_start, 4)
            profile_rows.append(profile)
            continue
            
        print(f"\n🔍 {form_type} Accession {i+1}: {acc} | Report or Filing Date: {report_date}")
        extracted = try_all_htm_files(cik, acc, headers, profile=profile, report_date=report_date)
        
        if not extracted:
            profile["elapsed_seconds"] = round(time.time() - accession_start, 4)
            profile_rows.append(profile)
            continue
        profile["success"] = True
        profile["result_count"] = len(extracted)
        for result in extracted:
            results.append({
                "accession": acc,
                "report_date": report_date,
                "file": result["file"],
                "url": result["url"],
                "document_period_end": result["document_period_end"],
                "document_period_label": result["document_period_label"],
                "facts": result["facts"],
                "context_blocks": result["context_blocks"],
                "concept_roles": result["concept_roles"],
                "form": form_type
            })

        profile["elapsed_seconds"] = round(time.time() - accession_start, 4)
        profile_rows.append(profile)

    form_key = form_type.lower().replace("-", "")
    elapsed_values = [row["elapsed_seconds"] for row in profile_rows]
    slowest_rows = sorted(profile_rows, key=lambda row: row["elapsed_seconds"], reverse=True)[:5]
    extraction_profile_summary = {
        "accession_total": len(profile_rows),
        "accession_success": sum(1 for row in profile_rows if row["success"]),
        "pre_2019_skipped": sum(1 for row in profile_rows if row["pre_2019_skipped"]),
        "elapsed_seconds_total": round(sum(elapsed_values), 4),
        "elapsed_seconds_p50": round(_compute_percentile(elapsed_values, 0.5) or 0.0, 4),
        "elapsed_seconds_p90": round(_compute_percentile(elapsed_values, 0.9) or 0.0, 4),
        "htm_candidates_total": sum(row.get("htm_candidates", 0) for row in profile_rows),
        "htm_attempts_total": sum(row.get("htm_attempts", 0) for row in profile_rows),
        "htm_attempt_seconds_total": round(sum(row.get("htm_attempt_seconds_total", 0.0) for row in profile_rows), 4),
        "fallback_scan_accessions": sum(1 for row in profile_rows if row.get("used_fallback_scan")),
        "slowest_accessions": [
            {
                "accession": row["accession"],
                "report_date": row["report_date"],
                "elapsed_seconds": row["elapsed_seconds"],
                "success": row["success"],
                "htm_candidates": row.get("htm_candidates", 0),
                "htm_attempts": row.get("htm_attempts", 0),
                "used_fallback_scan": row.get("used_fallback_scan", False),
                "selected_file": row.get("selected_file"),
                "selected_fact_count": row.get("selected_fact_count", 0),
                "inferred_period_end_used": row.get("inferred_period_end_used", False),
            }
            for row in slowest_rows
        ],
    }
    log_metric(f"extraction_profile_{form_key}", extraction_profile_summary)
    print(
        f"📈 {form_type} extraction profile: accessions={extraction_profile_summary['accession_total']} "
        f"success={extraction_profile_summary['accession_success']} "
        f"p50={extraction_profile_summary['elapsed_seconds_p50']:.2f}s "
        f"p90={extraction_profile_summary['elapsed_seconds_p90']:.2f}s "
        f"htm_attempts={extraction_profile_summary['htm_attempts_total']}"
    )
    return results


def parse_filing(
    ticker: str,
    year: int,
    quarter: int,
    full_year_mode: bool = False,
    debug_mode: bool = False,
) -> dict:
    """Parse and enrich filing data up to the extraction/matching boundary."""
    TICKER = ticker
    YEAR = year
    QUARTER = quarter
    FULL_YEAR_MODE = full_year_mode
    DEBUG_MODE = debug_mode
    EXCEL_FILE = None
    SHEET_NAME = None
    excel_enabled = False

    # Local mutable pull window values.
    N_10Q = N_10Q_BASE
    N_10K = N_10K_BASE

    # Explicit defaults for downstream return payload fields.
    target_label = None
    annual_label = None
    target_10q = None
    prior_10q = None
    target_10k = None
    prior_10k = None
    q1_entry = None
    q2_entry = None
    q3_entry = None
    q1_prior_entry = None
    q2_prior_entry = None
    q3_prior_entry = None
    df_current = None
    df_prior = None
    df_current_10k = None
    df_prior_10k = None
    df_q1 = None
    df_q2 = None
    df_q3 = None
    df_q1_prior = None
    df_q2_prior = None
    df_q3_prior = None
    negated_tags = set()

    # === Add inputs to metrics dictionary ===
    metrics.update({
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "full_year_mode": full_year_mode,
        "debug_mode": debug_mode,
        "start_time": datetime.now().isoformat()
    })


    # In[95]:


    # === EXTRACTION CONFIG ===

    CIK = lookup_cik_from_ticker(TICKER)
    FOUR_Q_MODE = (QUARTER == 4)  # 🆕 Build 4Q flag

    # Adjust number of filings to pull based on mode - # You might need more for 4Q builds (Q1–Q3 of both years)
    N_10Q = N_10Q + N_10Q_EXTRA
    N_10K = N_10K + N_10K_EXTRA

    # === Enforce quarter numbers == 

    if QUARTER not in [1, 2, 3, 4]:
        raise ValueError(f"❌ Invalid quarter value: {QUARTER}. Must be 1, 2, 3, or 4.")

    # === Enforce FULL_YEAR_MODE only when QUARTER == 4
    if QUARTER != 4:
        FULL_YEAR_MODE = False

    # === Check if Excel file exists ===
    if excel_enabled:
        if not os.path.exists(EXCEL_FILE):
            print(f"⚠️ Warning: Excel file '{EXCEL_FILE}' does not exist yet. It will need to be created or exported later.")
        else:
            print(f"✅ Excel file '{EXCEL_FILE}' found.")

    # === Check if CIK was properly loaded ===
    if not CIK:
        raise ValueError("❌ No valid CIK provided. Please set CIK or lookup from TICKER.")
    print(f"✅ Using CIK for {TICKER}: {CIK}")


    # Notes:
    # - Always run cells in sequence
    # - After export to Updater.xlsm, manually archive if needed
    # - Change CIK/YEAR/QUARTER above before rerunning


    # In[96]:


    # === FETCH & PARSE FILINGS ===================================
    # Function to parse filings and label the data
    # === Categorize Periods for Extracted Facts from Filing(s) ===

    # (imports already at module level: datetime, timedelta, etree, pd, re)


    # === Helper: Enrich a Filing ===


    # In[97]:


    # === FETCH & PARSE FILINGS ===================================
    # Fetches recent filings of company from JSON API and labels and puts 10K's and 10Q's in a list

    # === Fetch recent 10-Q and 10-K accessions from EDGAR ===


    # === FETCH 10Q/10K ACCESSIONS ===
    if ADAPTIVE_DISCOVERY:
        required_min_10q = 6
        required_min_10k = 3
    else:
        required_min_10q = N_10Q
        required_min_10k = N_10K

    accessions_10q, accessions_10k = fetch_recent_10q_10k_accessions(
        CIK,
        HEADERS,
        N_10Q,
        N_10K,
        min_10q=required_min_10q,
        min_10k=required_min_10k,
    )
    print(accessions_10q[:2], accessions_10k[:1]) #preview

    # === FILTER BY YEAR AND MAX FILINGS ===
    accessions_10q = filter_filings_by_year(accessions_10q, YEAR, N_10Q)
    accessions_10k = filter_filings_by_year(accessions_10k, YEAR, N_10K)

    # === If too few filings, fallback to full master index scan
    if len(accessions_10q) < required_min_10q or len(accessions_10k) < required_min_10k:
        print(f"⚠️ Not enough filings from recent submissions — falling back to master index.")
        use_fallback = True
        log_metric("fallback_triggered", True)
    else:
        use_fallback = False
        print("✅ Using recent submissions only — fallback not needed.")
        log_metric("fallback_triggered", False)


    # In[98]:


    # === FETCH & PARSE RECENT FILINGS ===================================
    # === Labels 10-Q accessions from EDGAR with fiscal-end dates and assign quarters ===

    # === LABEL 10Q ACCESSIONS ===
    accessions_10q = label_10q_accessions(accessions_10q, accessions_10k)

    if not FOUR_Q_MODE:
        # === Guard: Ensure labeled target quarter exists ===
        target_label = f"{QUARTER}Q{str(YEAR)[-2:]}"  # e.g., 2Q24
    
        # === Filter results_10q based on target label
        filtered_10qs = [q for q in accessions_10q if q.get("label") == target_label]
    
        if not filtered_10qs:
           raise FilingNotFoundError(f"❌ No 10-Q filing found for {target_label}. This quarter may not have been filed yet.")


    # In[99]:


    # === FETCH & PARSE RECENT FILINGS ===================================
    # === Filter for required 10-Q's for workflow ===================================

    required_10q_filings = []
    required_10k_filings = []
    fallback_10q_candidates = []
    fallback_10k_candidates = []

    # === FILTER FOR REQUIRED 10Q ACCESSIONS ===
    if not use_fallback:
        required_10q_filings = filter_10q_accessions(accessions_10q, YEAR, QUARTER)


    # In[100]:


    # === FETCH & PARSE RECENT FILINGS ===================================
    # === Label 10K's with fiscal year data ===================================

    # === ENRICH 10K ACCESSIONS WITH FISCAL YEAR METADATA ===

    accessions_10k = enrich_10k_accessions_with_fiscal_year(accessions_10k)

    # === Guard: Ensure target 10-K year is present ===

    if FOUR_Q_MODE:
        annual_label = f"FY{str(YEAR)[-2:]}"  # Example: "FY24"
        print(f"\n🎯 Annual Label: {annual_label}")
    
    # Select current year 10-K
        filtered_10ks = [k for k in accessions_10k if k.get("year") == YEAR]

        if not filtered_10ks:
            raise FilingNotFoundError(f"❌ No matching 10-K found for {YEAR}. It may not have been filed yet.")    


    # In[101]:


    # === FETCH & PARSE RECENT FILINGS ===================================
    # === Filter for required 10-K filings ===================================

    # === FILTER FOR REQUIRED 10Q ACCESSIONS ===
    if not use_fallback:
        required_10k_filings = filter_10k_accessions(accessions_10k, YEAR, QUARTER)


    # In[102]:


    # === FALLBACK: FETCH & PARSE FILINGS ===================================
    # Looks into the master index to find the 10K's and 10Q's and puts them in a list

    # (imports already at module level: requests, gzip, BytesIO, datetime)

    # === Calculate required year window for master index lookup
    YEARS_TO_PULL = N_10K  # Based on N_10K = 4 and N_10Q = 12
    years_to_check = list(range(YEAR - (YEARS_TO_PULL - 1), YEAR + 2))  # [2020, 2021, 2022, 2023, 2024] for 2023
    quarters_to_check = ["QTR1", "QTR2", "QTR3", "QTR4"]

    # === Fetch 10-Q and 10-K accessions from EDGAR master index ===

    # === FETCH 10Q/10K ACCESSIONS FROM MASTER INDEX ===

    if use_fallback:
        accessions_10q, accessions_10k = fetch_10q_10k_accessions_from_master(CIK, HEADERS, years_to_check, quarters_to_check)
        print(accessions_10q[:2], accessions_10k[:1])

        # Recompute labels on complete master-index lists, then narrow candidates.
        accessions_10q = label_10q_accessions(accessions_10q, accessions_10k)
        accessions_10k = enrich_10k_accessions_with_fiscal_year(accessions_10k)

        fallback_10q_candidates = filter_10q_accessions(accessions_10q, YEAR, QUARTER)
        fallback_10k_candidates = filter_10k_accessions(accessions_10k, YEAR, QUARTER)

        # Add a bounded recency buffer so mislabeling does not force immediate full-list widening.
        fallback_10q_cap = max(required_min_10q + 4, 10)
        fallback_10k_cap = max(required_min_10k + 2, 5)
        fallback_10q_candidates = _merge_entries_by_accession(
            fallback_10q_candidates,
            accessions_10q,
            max_items=fallback_10q_cap,
        )
        fallback_10k_candidates = _merge_entries_by_accession(
            fallback_10k_candidates,
            accessions_10k,
            max_items=fallback_10k_cap,
        )
        print(
            f"📌 Fallback candidate set sizes: {len(fallback_10q_candidates)} 10-Q, "
            f"{len(fallback_10k_candidates)} 10-K"
        )


    # In[103]:


    # === FETCH & PARSE FILINGS ===================================
    # Extracts data from filings starting from the target filing from CONFIG to use
    # === Extract facts from 10-Q and 10-K accessions from EDGAR ===

    start_total = time.time()

    # (imports already at module level: time, BeautifulSoup, XMLParsedAsHTMLWarning, warnings)
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    # === CONFIG ===
    STOP_AFTER_FIRST_VALID_PERIOD = True

    # === Extract facts and DocumentPeriodEndDate from a single .htm ===

    # === Try all .htm files inside an accession (starts with largest file, then stop after first valid) ===
    
    # === Extract information from list of 10K and 10Q filings ===

    # === EXTRACT INFORMATION FROM 10-Qs and 10-K's ===

    if not use_fallback:
        print("\n📘 Processing 10-Qs...")
        results_10q = extract_filing_batch(required_10q_filings, CIK, HEADERS, "10-Q")

        print("\n📕 Processing 10-Ks...")
        results_10k = extract_filing_batch(required_10k_filings, CIK, HEADERS, "10-K")
    
    else:
        print("\n⚠️ Fallback mode: parsing narrowed candidate lists first.")

        print("\n📘 Processing 10-Q candidate subset...")
        results_10q = extract_filing_batch(fallback_10q_candidates, CIK, HEADERS, "10-Q")

        print("\n📕 Processing 10-K candidate subset...")
        results_10k = extract_filing_batch(fallback_10k_candidates, CIK, HEADERS, form_type="10-K")

        coverage_ok, coverage_reason = _fallback_coverage_satisfied(
            results_10q,
            results_10k,
            YEAR,
            QUARTER,
            FOUR_Q_MODE,
        )
        if coverage_ok:
            print(f"✅ Fallback subset satisfied coverage checks ({coverage_reason}).")
        else:
            print(f"⚠️ Fallback subset incomplete ({coverage_reason}) — widening to remaining filings.")
            remaining_10q = _remaining_entries(accessions_10q, fallback_10q_candidates)
            remaining_10k = _remaining_entries(accessions_10k, fallback_10k_candidates)

            if remaining_10q:
                print("\n📘 Processing remaining fallback 10-Qs...")
                results_10q.extend(extract_filing_batch(remaining_10q, CIK, HEADERS, "10-Q"))
            if remaining_10k:
                print("\n📕 Processing remaining fallback 10-Ks...")
                results_10k.extend(extract_filing_batch(remaining_10k, CIK, HEADERS, form_type="10-K"))

        results_10q = _sort_results_by_accession_order(results_10q, accessions_10q)
        results_10k = _sort_results_by_accession_order(results_10k, accessions_10k)

    # === CALCULATING PROCESSING TIME ===

    print("🎯 All filings processed. Script complete!")
    end_total = time.time()
    print(f"\n⏱️ Total extraction processing time: {end_total - start_total:.2f} seconds")
    log_metric("extraction_processing_seconds", round(end_total - start_total, 2))


    # In[104]:


    # === FETCH & PARSE FILINGS ===================================
    # === Labels 10-Q accessions from EDGAR with fiscal-end dates and assign quarters ===

    # === Extract and sort valid fiscal year-end dates from 10-Ks ===
    fiscal_year_ends = []

    for entry in results_10k:
        fy_date = parse_date(entry["document_period_end"])
        if fy_date:
            fiscal_year_ends.append(fy_date)

    fiscal_year_ends = sorted(fiscal_year_ends, reverse=True)

    if not fiscal_year_ends:
        raise ValueError("No valid fiscal year-end dates found in 10-Ks.")

    # === Match each 10-Q to its fiscal year and assign quarter label ===
    print("\n📊 Matching 10-Qs to fiscal year-end and labeling quarters based off report date:")

    for q in results_10q:
        q_date = parse_date(q["document_period_end"])
        if not q_date:
            q["quarter"] = None
            q["label"] = None
            continue

        # Match to the fiscal year that this 10-Q falls into — first fiscal year-end after Q end
    
        # Prefer fiscal year-ends >= Q date (standard case)
        candidates = [fy for fy in fiscal_year_ends if fy >= q_date]
    
        if candidates:
            matched_fy = min(candidates)
            used_fallback = False
        else:
            # Fallback: use latest fiscal year-end before Q date
            candidates = [fy for fy in fiscal_year_ends if fy < q_date]
            matched_fy = max(candidates) if candidates else None
            used_fallback = True
    
        # 🧠 Shift forward if using fallback (e.g., using FY23 to label FY24)
        if matched_fy and used_fallback:
            matched_fy = matched_fy.replace(year=matched_fy.year + 1)

        if not matched_fy:
            print(f"⚠️ No matching fiscal year-end found for 10-Q ending {q['document_period_end']}")
            q["quarter"] = None
            q["label"] = None
            continue

        # Use day-based logic to assign correct quarter
        days_diff = (matched_fy - q_date).days

        if 70 <= days_diff <= 120:
            quarter = "Q3"
        elif 160 <= days_diff <= 200:
            quarter = "Q2"
        elif 250 <= days_diff <= 300:
            quarter = "Q1"
        else:
            print(f"⚠️ Unexpected delta ({days_diff} days) between {matched_fy.strftime('%Y-%m-%d')} and {q_date.strftime('%Y-%m-%d')} — nonstandard quarter")
            q["quarter"] = None
            q["label"] = None
            q["non_standard_period"] = True
            continue

        # Apply labels to the quarterly filings
        q["fiscal_year_end"] = matched_fy
        q["quarter"] = quarter
        q["year"] = q_date.year    #Note: Fiscal year
        q["label"] = f"{quarter[1:]}Q{str(matched_fy.year)[-2:]}"  # e.g. "Q1" + "25" → "1Q25" #Uses fiscal year match

        print(f"✅ {q['document_period_end']} → {q['label']} (matched FY end {q['fiscal_year_end']})")


    # In[105]:


    # === FETCH & PARSE FILINGS ===================================
    # Gathers fiscal year end date from recent 10-K filings
    # === Enrich 10-K results with fiscal year metadata ===

    print("\n🛠 Enriching 10-Ks with fiscal year and fiscal year-end...")

    for k in results_10k:
        period_end = k.get("document_period_end")
        dt = parse_date(period_end)

        if dt:
            k["year"] = dt.year
            k["fiscal_year_end"] = dt
            print(f"✅ {period_end} → Fiscal Year {k['year']} | Fiscal Year End: {k['fiscal_year_end']}")
        else:
            k["year"] = None
            k["fiscal_year_end"] = None
            print(f"⚠️ Could not parse period end for accession {k['accession']}")


    # In[106]:


    # === NORMAL 10-Q WORKFLOW ====================================
    #Identifies the target 10-Q filing to use
    # === Filter 10-Qs Based on YEAR and QUARTER from CONFIG for Target Filings ===

    # Build target label from your existing config
    target_label = f"{QUARTER}Q{str(YEAR)[-2:]}"  # e.g., 2Q24
    print(f"\n🎯 Target Label: {target_label}")

    if FOUR_Q_MODE:
        print("📄 4Q mode detected: will select 10-K filing and prior 10-Q's instead of specific 10-Q.")
        target_10q = None
  
    else:
        # Normal flow: Pick 10-Q
        # === Filter results_10q based on target label
        filtered_10qs = [q for q in results_10q if q.get("label") == target_label]
    
        # === Output matching 10-Qs
        print(f"\n📄 Matching 10-Qs for {target_label}:")
    
        if not filtered_10qs:
            target_10q = None
            raise FilingNotFoundError(f"❌ No 10-Q filing found for {target_label}. This quarter may not have been filed yet.")
        
        else:
            target_10q = filtered_10qs[0]
            for q in filtered_10qs:
                print(f"✅ {q['label']} | Period End: {q['document_period_end']} | URL: {q['url']}")

    # === Log Target 10-Q ===
    if not FOUR_Q_MODE and target_10q:
        log_metric("target_filing", {
            "type": "10-Q",
            "label": target_10q["label"],
            "period_end": target_10q["document_period_end"],
            "url": target_10q["url"]
        })


    # In[107]:


    # === 4Q WORKFLOW =============================================
    # Identifies the target 10-K and prior 10-K, and current year Q3 10-Q

    # Build annual label from your existing config
    annual_label = f"FY{str(YEAR)[-2:]}"  # Example: "FY24"
    print(f"\n🎯 Annual Label: {annual_label}")

    if FOUR_Q_MODE:
    # Select current year 10-K
        filtered_10ks = [k for k in results_10k if k.get("year") == YEAR]
    
        if not filtered_10ks:
            raise FilingNotFoundError(f"❌ No matching 10-K found for {YEAR}. It may not have been filed yet.")

        target_10k = filtered_10ks[0]
        print(f"Selected 10-K for full year: Period-End: {target_10k['document_period_end']}")
        print(f"URL: {target_10k['url']}")

        # Use fiscal year end from selected 10-K. 
        # Note: fiscal_year_end is in YYYY-MM-DD string format (assigned during quarter labeling step)
    
        fye_target = target_10k["fiscal_year_end"]
    
        # Select current year Q1–Q3 10-Qs by fiscal year end
        q1_entry = next((q for q in results_10q if q.get("quarter") == "Q1" and q.get("fiscal_year_end") == fye_target), None)
        q2_entry = next((q for q in results_10q if q.get("quarter") == "Q2" and q.get("fiscal_year_end") == fye_target), None)
        q3_entry = next((q for q in results_10q if q.get("quarter") == "Q3" and q.get("fiscal_year_end") == fye_target), None)

        # === Store Quarter Entries in a Dict ===
        quarter_entries = {
            "Q1": q1_entry,
            "Q2": q2_entry,
            "Q3": q3_entry
        }

        # Check for missing
        missing_qs = []
        if not q1_entry: missing_qs.append("Q1")
        if not q2_entry: missing_qs.append("Q2")
        if not q3_entry: missing_qs.append("Q3")

        if missing_qs:
            print(f"\n⚠️ Missing current year 10-Qs for: {', '.join(missing_qs)}")
            if "Q1" in missing_qs: q1_entry = None
            if "Q2" in missing_qs: q2_entry = None
            if "Q3" in missing_qs: q3_entry = None

        print(f"\n✅ Found Q1-Q3 10-Qs for fiscal year {YEAR}:")

        for q, entry in quarter_entries.items():
            if entry and "document_period_end" in entry and "url" in entry:
                print(f"   -{q}: Period End: {entry['document_period_end']} | {entry['url']}")

        if not q3_entry:
            raise FilingNotFoundError("❌ Missing current year Q3 10-Q — required for 4Q processing.")
    
        # Select prior year 10-K - this may be redundant - used in next step
        prior_10k = next((k for k in results_10k if k.get("year") == YEAR - 1), None)
    
        if not prior_10k and not FULL_YEAR_MODE:
            raise FilingNotFoundError(f"❌ No matching 10-K found for {YEAR} — required for prior 4Q calculation.")

        elif not prior_10k and FULL_YEAR_MODE:
            print(f"⚠️ Missing prior year 10-K for {YEAR - 1}, but continuing to process filing.")

        if prior_10k:
            print(f"\n✅ Selected prior 10-K: Period-End: {prior_10k['document_period_end']}")
            print(f"URL: {prior_10k['url']}")
        
        # === Log Target 10-K ===
        log_metric("target_filing", {
            "type": "10-K",
            "label": annual_label,
            "period_end": target_10k["document_period_end"],
            "url": target_10k["url"]
        })   

    else:
        target_10k = None
        q1_entry = None
        q2_entry = None
        q3_entry = None
        prior_10k = None


    # In[108]:


    # === NORMAL 10-Q WORKFLOW and 4Q WORKFLOW =============================================
    # Identifies the prior quarterly and annual filings to use 

    # === Normal mode (e.g., 1Q, 2Q, or 3Q)
    # === STEP: Find prior y/y Q filings from results_10q

    if not FOUR_Q_MODE:

        quarter = target_10q.get("quarter")
        fye_str = target_10q.get("fiscal_year_end")  # already in 'YYYY-MM-DD' format

        # Identify previous fiscal year-end from known values (sorted descending)
        fiscal_ends = sorted({q.get("fiscal_year_end") for q in results_10q if q.get("fiscal_year_end")}, reverse=True)
        try:
            idx = fiscal_ends.index(fye_str)
            prior_fye_str = fiscal_ends[idx + 1]  # next fiscal year end in time
        except (ValueError, IndexError):
            prior_fye_str = None

        prior_10q = None
        if prior_fye_str:
            prior_10q = next(
                (q for q in results_10q if q.get("quarter") == quarter and q.get("fiscal_year_end") == prior_fye_str),
                None
            )
    
        if prior_10q:
            print(f"\n✅ Found prior 10-Q: {prior_10q['label']}")
            print(f"Period End: {prior_10q['document_period_end']}")
            print(f"URL: {prior_10q['url']}")
        else:  
            print(f"\n⚠️ Could not find prior 10-Q.")

    else:
    
    # === 4Q mode: find prior 10-K and prior Q3 10-Qs
        prior_10k = next((k for k in results_10k if k.get("year") == (YEAR - 1)), None)
    
        if prior_10k:
            print(f"\n✅ Found prior 10-K for {YEAR-1}:")
            print(f"Period End: {prior_10k['document_period_end']}")
            print(f"URL: {prior_10k['url']}")
        else:
            print(f"\n⚠️ Could not find prior 10-K.")

        # Use the prior 10-K document period end as fiscal year end anchor
        fye_prior = prior_10k["fiscal_year_end"] if prior_10k else None

        # Match prior 10-Qs with the same fiscal year end and correct quarter
        q1_prior_entry = next((q for q in results_10q if q.get("quarter") == "Q1" and q.get("fiscal_year_end") == fye_prior), None)
        q2_prior_entry = next((q for q in results_10q if q.get("quarter") == "Q2" and q.get("fiscal_year_end") == fye_prior), None)
        q3_prior_entry = next((q for q in results_10q if q.get("quarter") == "Q3" and q.get("fiscal_year_end") == fye_prior), None)

        # === Store Prior Quarter Entries in a Dict ===
        prior_quarter_entries = {
            "Q1": q1_prior_entry,
            "Q2": q2_prior_entry,
            "Q3": q3_prior_entry
        }

        if not FULL_YEAR_MODE:
            print(f"\n✅ Prior 10-Qs found for fiscal year {YEAR - 1}:")

        for q_label, q_entry in prior_quarter_entries.items():
            if q_entry:
                print(f"  - {q_label}: Period End: {q_entry['document_period_end']} | URL: {q_entry['url']}")

        if not q3_prior_entry:
            raise ValueError("❌ Missing prior year Q3 10-Q — required for 4Q processing.")

        # Check for missing entries
        missing_prior_qs = []
        if not q1_prior_entry: missing_prior_qs.append("Q1")
        if not q2_prior_entry: missing_prior_qs.append("Q2")
        if not q3_prior_entry: missing_prior_qs.append("Q3")

        if missing_prior_qs:
            print(f"\n⚠️ Missing current year 10-Qs for: {', '.join(missing_prior_qs)}")
            if "Q1" in missing_prior_qs: q1_prior_entry = None
            if "Q2" in missing_prior_qs: q2_prior_entry = None
            if "Q3" in missing_prior_qs: q3_prior_entry = None

        if FULL_YEAR_MODE:
            print("⚠️ Skipping prior Q1–Q3 10-Q check — not needed in full-year mode.")


    # In[109]:


    # === SHARED LOGIC (e.g. negated labels, exports) =============
    # === Get Negated Labels (for Visual Presentation) ===

    if FOUR_Q_MODE:
        if target_10k is None:
            raise ValueError("❌ target_10k is None — check 10-K selection.")
        negated_tags = get_negated_label_concepts(CIK, target_10k["accession"], HEADERS)

    else:
        if target_10q is None:
            raise ValueError("❌ target_10q is None — check 10-Q selection.")
        negated_tags = get_negated_label_concepts(CIK, target_10q["accession"], HEADERS)


    # In[110]:


    # === SHARED LOGIC (e.g. negated labels, exports) =============
    # === Extract Concept Roles from .pre.xml for mapping ===

    if FOUR_Q_MODE:
        if target_10k is None:
            raise ValueError("❌ target_10k is None — check 10-K selection.")
        concept_roles = get_concept_roles_from_presentation(CIK, target_10k["accession"], HEADERS)
        filename_roles_export = f"{CIK}_{annual_label}_presentation_roles.csv"

    else:
        if target_10q is None:
            raise ValueError("❌ target_10q is None — check 10-Q selection.")
        concept_roles = get_concept_roles_from_presentation(CIK, target_10q["accession"], HEADERS)
        filename_roles_export = f"{CIK}_{target_label}_presentation_roles.csv"

    # Convert to DataFrame
    rows = []
    for tag, roles in concept_roles.items():
        for role in roles:
            rows.append({"tag": tag, "presentation_role": role})

    df_concept_roles = pd.DataFrame(rows)

    # Preview
    print(f"✅ Extracted {len(df_concept_roles)} concept→role entries from .pre.xml")
    log_metric("concept_roles_extracted", len(df_concept_roles))


    # In[111]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # Enrich target 10-Q and prior 10-Q data with labels
 
    if not FOUR_Q_MODE:
        print("\n📥 Enriching normal 10-Q mode...")

        # Enrich current 10-Q

        df_current = enrich_filing(target_10q, results_10q, results_10k)
        print(df_current["matched_category"].value_counts(dropna=False))
        categorized_Q_fact_counts = df_current["matched_category"].value_counts(dropna=False).to_dict()
    
        # Enrich prior 10-Q

        if prior_10q is None:
            raise ValueError(f"❌ No prior 10-Q found for {TICKER} {QUARTER}Q{YEAR} — cannot proceed without prior comparison.")
    
        else:
            df_prior = enrich_filing(prior_10q, results_10q, results_10k)

        print(df_prior["matched_category"].value_counts(dropna=False))
        log_metric("fact_category_counts", categorized_Q_fact_counts)


    # In[112]:


    # === 4Q WORKFLOW =============================================
    # Enrich target 10-K's and prior 10-Q's data with labels (for 4Q calculations)

    if FOUR_Q_MODE:
        print("\n📥 Enriching 4Q mode (10-K + Prior 10-Q's...)")

        # Current year
        df_current_10k = enrich_filing(target_10k, results_10q, results_10k)
        print("🔹 Current Year 10-K facts enriched:")
        print(df_current_10k["matched_category"].value_counts(dropna=False))
        categorized_K_fact_counts = df_current_10k["matched_category"].value_counts(dropna=False).to_dict()

        df_q1 = df_q2 = df_q3 = None
        if q1_entry:
            df_q1 = enrich_filing(q1_entry, results_10q, results_10k)
            print("🔹 Current Year Q1 facts enriched:")
            print(df_q1["matched_category"].value_counts(dropna=False))

        if q2_entry:
            df_q2 = enrich_filing(q2_entry, results_10q, results_10k)
            print("🔹 Current Year Q2 facts enriched:")
            print(df_q2["matched_category"].value_counts(dropna=False))

        if q3_entry:
            df_q3 = enrich_filing(q3_entry, results_10q, results_10k)
            print("🔹 Current Year Q3 facts enriched:")
            print(df_q3["matched_category"].value_counts(dropna=False))

        # Prior year
        if prior_10k:
            df_prior_10k = enrich_filing(prior_10k, results_10q, results_10k)
            print("🔹 Prior Year 10-K facts enriched:")
            print(df_prior_10k["matched_category"].value_counts(dropna=False))
        else:
            df_prior_10k = pd.DataFrame() # allow df_prior_10k to be created to allow FY downstream workflow

        df_q1_prior = df_q2_prior = df_q3_prior = None
        if q1_prior_entry:
            df_q1_prior = enrich_filing(q1_prior_entry, results_10q, results_10k)
            print("🔹 Prior Year Q1 facts enriched:")
            print(df_q1_prior["matched_category"].value_counts(dropna=False))

        if q2_prior_entry:
            df_q2_prior = enrich_filing(q2_prior_entry, results_10q, results_10k)
            print("🔹 Prior Year Q2 facts enriched:")
            print(df_q2_prior["matched_category"].value_counts(dropna=False))

        if q3_prior_entry:
            df_q3_prior = enrich_filing(q3_prior_entry, results_10q, results_10k)
            print("🔹 Prior Year Q3 facts enriched:")
            print(df_q3_prior["matched_category"].value_counts(dropna=False))

        log_metric("fact_category_counts", categorized_K_fact_counts)

    else:
        pass


    # In[113]:


    # === SHARED LOGIC (e.g. negated labels, exports) =============
    # === Check Negated Labels ===

    # Ensure negated_tags is a set
    negated_list = sorted(list(negated_tags))

    # Create a DataFrame
    df_negated_labels = pd.DataFrame(negated_list, columns=["tag_with_negated_label"])

    # Preview in notebook
    print(f"✅ Found {len(df_negated_labels)} tags with negated labels in .pre.xml")
    log_metric("negated_labels_extracted", len(df_negated_labels))


    # In[114]:


    # === SHARED LOGIC (Enrichment summary) =============
    #Preview of the enrichment results for target current and prior year filing with export to review

    if FOUR_Q_MODE:
        print(f"📄 Extracted facts from target: {target_10k.get('form', 'Unknown')} ending {target_10k.get('document_period_end', 'Unknown')}")
        print(f"✅ Fact categorization summary (logged): {categorized_K_fact_counts}")

    else:
        print(f"📄 Extracted facts from target: {target_10q.get('form', 'Unknown')} ending {target_10q.get('document_period_end', 'Unknown')}")
        print(f"✅ Fact categorization summary (logged): {categorized_Q_fact_counts}")
    


    # In[115]:


    # === 4Q WORKFLOW =============================================
    # === CLEAN AXIS VALUES FIRST ===
    if FOUR_Q_MODE:
    
        for col in AXIS_COLS:
            df_current_10k[col] = df_current_10k[col].fillna("__NONE__")
            df_prior_10k[col] = df_prior_10k[col].fillna("__NONE__")
            df_q3[col] = df_q3[col].fillna("__NONE__")
            df_q3_prior[col] = df_q3_prior[col].fillna("__NONE__")

    mode = "FY" if FOUR_Q_MODE and FULL_YEAR_MODE else ("4Q" if FOUR_Q_MODE else "Q")
    return {
        "mode": mode,
        "ticker": TICKER,
        "year": YEAR,
        "quarter": QUARTER,
        "full_year_mode": FULL_YEAR_MODE,
        "debug_mode": DEBUG_MODE,
        "FOUR_Q_MODE": FOUR_Q_MODE,
        "CIK": CIK,
        "target_label": target_label,
        "annual_label": annual_label,
        "target_10q": target_10q,
        "prior_10q": prior_10q,
        "target_10k": target_10k,
        "prior_10k": prior_10k,
        "q1_entry": q1_entry,
        "q2_entry": q2_entry,
        "q3_entry": q3_entry,
        "q1_prior_entry": q1_prior_entry,
        "q2_prior_entry": q2_prior_entry,
        "q3_prior_entry": q3_prior_entry,
        "df_current": df_current,
        "df_prior": df_prior,
        "df_current_10k": df_current_10k,
        "df_prior_10k": df_prior_10k,
        "df_q1": df_q1,
        "df_q2": df_q2,
        "df_q3": df_q3,
        "df_q1_prior": df_q1_prior,
        "df_q2_prior": df_q2_prior,
        "df_q3_prior": df_q3_prior,
        "negated_tags": set(negated_tags),
        "metrics": dict(metrics),
    }
