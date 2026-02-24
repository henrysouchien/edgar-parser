"""Period matching layer for parsed EDGAR filings."""

import numpy as np
import pandas as pd

from .utils import (
    AXIS_COLS,
    FINAL_COLS,
    audit_value_collisions,
    log_metric,
    parse_date,
    run_adaptive_match_keys,
    standardize_zip_output,
    zip_match_in_order,
)


def apply_visual_signs(df, negated_tags):
    
    """
    Applies visual sign flipping to financial values for tags that are conceptually positive
    (e.g., expenses, losses) but reported as negative in XBRL.

    This function creates two new columns—`visual_current_value` and `visual_prior_value`—by reversing
    the sign of values for tags that appear in the `negated_tags` set. This transformation is useful
    for display, analysis, or charting where expenses should be shown as positive values.

    Args:
        df (pandas.DataFrame): DataFrame containing at least the following columns:
            - 'tag': XBRL tag name (e.g., "us-gaap:OperatingExpenses")
            - 'current_period_value': Numeric value for the current period
            - 'prior_period_value': Numeric value for the prior period
        negated_tags (set or list): Set of tags that should be visually flipped.

    Returns:
        pandas.DataFrame: Same DataFrame with two new columns:
            - 'visual_current_value': current_period_value (sign flipped if tag in negated_tags)
            - 'visual_prior_value': prior_period_value (sign flipped if tag in negated_tags)

    Example:
        df = apply_visual_signs(df, {"us-gaap:OperatingExpenses", "us-gaap:LossContingency"})
    """
    
    df["visual_current_value"] = df.apply(
        lambda row: -row["current_period_value"] if row["tag"] in negated_tags and pd.notna(row["current_period_value"]) else row["current_period_value"],
        axis=1
    )
    df["visual_prior_value"] = df.apply(
        lambda row: -row["prior_period_value"] if row["tag"] in negated_tags and pd.notna(row["prior_period_value"]) else row["prior_period_value"],
        axis=1
    )
    return df


def _apply_collision_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Apply collision flags based on duplicate prior/current value pairings."""
    flagged_df = audit_value_collisions(df)
    flagged_values = set(flagged_df["current_period_value"])
    df = df.copy()
    df["collision_flag"] = df["current_period_value"].apply(
        lambda x: 1 if x in flagged_values else 0
    )

    num_total_rows = len(df)
    num_unique_collisions = flagged_df[["tag", "current_period_value", "prior_period_value"]].drop_duplicates().shape[0]
    collision_rate = num_unique_collisions / max(num_total_rows, 1)
    print(f"🔁 Unique (tag, current, prior) collision combinations: {num_unique_collisions}")
    print(f"📊 Collision rate: {collision_rate:.1%}")
    log_metric("collision_rate", collision_rate)
    return df


def _finalize_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Final, shared export cleanup."""
    df = df.dropna(how="all")
    df = df[df["tag"].notna()]
    df = df[df["tag"].str.strip() != ""]
    df = df.dropna(subset=["current_period_value", "prior_period_value"], how="all")
    df = df.drop_duplicates(subset=["current_period_value", "prior_period_value"])
    return df.reset_index(drop=True)


def _match_4q_common(parsed: dict, full_year_mode: bool):
    FOUR_Q_MODE = True
    FULL_YEAR_MODE = full_year_mode

    df_current_10k = parsed.get("df_current_10k")
    df_prior_10k = parsed.get("df_prior_10k")
    df_q1 = parsed.get("df_q1")
    df_q2 = parsed.get("df_q2")
    df_q3 = parsed.get("df_q3")
    df_q1_prior = parsed.get("df_q1_prior")
    df_q2_prior = parsed.get("df_q2_prior")
    df_q3_prior = parsed.get("df_q3_prior")

    if FOUR_Q_MODE:
    
        MATCH_COLS_FY = ["tag", "date_type"] + AXIS_COLS
        MIN_MATCH_COLS_FY = ["tag", "date_type"]
    
        # Pull both from current 10-K
        df_fy_curr = df_current_10k[df_current_10k["matched_category"] == "current_full_year"].copy()
        df_fy_prior = df_current_10k[df_current_10k["matched_category"] == "prior_full_year"].copy()

        for col in AXIS_COLS:
            df_fy_curr[col] = df_fy_curr[col].fillna("__NONE__")
            df_fy_prior[col] = df_fy_prior[col].fillna("__NONE__")

        # Adaptive match
        match_keys_fy = run_adaptive_match_keys(df_fy_curr, df_fy_prior, MATCH_COLS_FY, MIN_MATCH_COLS_FY)
        df_fy_matched = zip_match_in_order(df_fy_curr, df_fy_prior, match_keys_fy)
        df_fy_matched = standardize_zip_output(df_fy_matched)
    
        # Preview results
        total_fy_curr = len(df_fy_curr)
        matched_fy_rows = len(df_fy_matched)
        match_rate_fy = matched_fy_rows / max(total_fy_curr, 1)
    
        print(f"✅ FY match rate: {match_rate_fy * 100:.1f}%")
        log_metric("match_rate", {"fy": match_rate_fy})



    # In[117]:


    # === 4Q WORKFLOW =============================================
    # === MATCH YTD (from Q3 10-Q in the fiscal year): current_ytd vs prior_ytd ===
    if FOUR_Q_MODE:

        MATCH_COLS_YTD = ["tag", "date_type"] + AXIS_COLS
        MIN_MATCH_COLS_YTD = ["tag", "date_type"]

        # Pull both from the same Q3 filing
        df_ytd_curr = df_q3[df_q3["matched_category"] == "current_ytd"].copy()
        df_ytd_prior = df_q3[df_q3["matched_category"] == "prior_ytd"].copy()

        # Ensure axis consistency
        for col in AXIS_COLS:
            df_ytd_curr[col] = df_ytd_curr[col].fillna("__NONE__")
            df_ytd_prior[col] = df_ytd_prior[col].fillna("__NONE__")

        # No need to shift 'end' — both from same Q3 filing
        match_keys_ytd = run_adaptive_match_keys(df_ytd_curr, df_ytd_prior, MATCH_COLS_YTD, MIN_MATCH_COLS_YTD)
        df_ytd_matched = zip_match_in_order(df_ytd_curr, df_ytd_prior, match_keys_ytd)
        df_ytd_matched = standardize_zip_output(df_ytd_matched)

        # Preview results
        total_ytd_curr = len(df_ytd_curr)
        matched_ytd_rows = len(df_ytd_matched)
        match_rate_ytd = matched_ytd_rows / max(total_ytd_curr, 1)

        print(f"✅ FY match rate: {match_rate_ytd * 100:.1f}%")
        log_metric("match_rate", {"ytd": match_rate_ytd})


    # In[118]:


    # === 4Q WORKFLOW =============================================
    # === CHECK FOR DUPLICATE MERGE KEYS AND FULL DUPLICATE ROWS ===

    if FOUR_Q_MODE:

        merge_key = ["tag", "date_type"] + AXIS_COLS

        # Recalculate _key
        df_fy_matched["_key"] = df_fy_matched[merge_key].apply(tuple, axis=1)
        df_ytd_matched["_key"] = df_ytd_matched[merge_key].apply(tuple, axis=1)

        # Count duplicate keys (rows with same _key)
        dupes_fy_keys = df_fy_matched["_key"].duplicated(keep=False).sum()
        dupes_ytd_keys = df_ytd_matched["_key"].duplicated(keep=False).sum()

        # Count unique keys and shared
        keys_fy = set(df_fy_matched["_key"])
        keys_ytd = set(df_ytd_matched["_key"])
        shared_keys = keys_fy & keys_ytd

        # Check full row duplicates
        full_dupes_fy = df_fy_matched.duplicated(keep=False).sum()
        full_dupes_ytd = df_ytd_matched.duplicated(keep=False).sum()

        print("🔍 Duplicate Key & Row Check:")
        print(f"  • FY matched: {dupes_fy_keys} rows with duplicate _key") 
        print(f"  • YTD matched: {dupes_ytd_keys} rows with duplicate _key")
        print(f"  • Unique keys in FY matched: {len(keys_fy)}")
        print(f"  • Unique keys in YTD matched: {len(keys_ytd)}")
        print(f"  • Shared keys between FY and YTD: {len(shared_keys)}")
        print("—")
        print(f"  • FY matched: {full_dupes_fy} fully duplicated rows")
        print(f"  • YTD matched: {full_dupes_ytd} fully duplicated rows")

        if FOUR_Q_MODE:
        
            # Identify rows with duplicated _key
            is_key_duplicated_fy = df_fy_matched["_key"].duplicated(keep=False)
            is_key_duplicated_ytd = df_ytd_matched["_key"].duplicated(keep=False)
    
            # Identify full-row duplicates
            is_full_dup_fy = df_fy_matched.duplicated(keep=False)
            is_full_dup_ytd = df_ytd_matched.duplicated(keep=False)
    
            # Check overlap
            key_and_full_fy = (is_key_duplicated_fy & is_full_dup_fy).sum()
            key_and_full_ytd = (is_key_duplicated_ytd & is_full_dup_ytd).sum()
    
            print("\n🔎 Duplicate Overlap Check:")
            print(f"  • FY matched: {key_and_full_fy} rows are both _key duplicates AND full duplicates")
            print(f"  • YTD matched: {key_and_full_ytd} rows are both _key duplicates AND full duplicates")
            print("\n✅ If these numbers equal the key duplicate count, it's safe to drop.")

            # Show rows with same _key but not fully duplicated
            non_exact_dupes_ytd = df_ytd_matched[
                df_ytd_matched["_key"].duplicated(keep=False) & ~df_ytd_matched.duplicated(keep=False)
            ]


    # In[119]:


    # === 4Q WORKFLOW =============================================
    # === CALCULATE 4Q: From Matched FY and YTD using aligned suffixes ===
    if FOUR_Q_MODE:

        merge_key = ["tag", "start_current"] + AXIS_COLS

        # Create merged key column to use for matching
        df_fy_matched["_key"] = df_fy_matched[merge_key].apply(tuple, axis=1)
        df_ytd_matched["_key"] = df_ytd_matched[merge_key].apply(tuple, axis=1)

        # Drop duplicates to prevent cartesian matching 
        df_fy_matched = df_fy_matched.drop_duplicates()
        df_ytd_matched = df_ytd_matched.drop_duplicates()

        # Safe merge (1:1 expected now)
        df_merged = pd.merge(
            df_fy_matched,
            df_ytd_matched[["_key", "current_period_value", "prior_period_value"]],
            on="_key",
            suffixes=("_current", "_prior")
        )

        # === Calculate 4Q values ===
        df_merged["current_period_value"] = df_merged["current_period_value_current"] - df_merged["current_period_value_prior"]
        df_merged["prior_period_value"] = df_merged["prior_period_value_current"] - df_merged["prior_period_value_prior"]

        # === Construct df_4q_output with correct names ===
        df_4q_output = pd.DataFrame()
        df_4q_output["current_tag"] = df_merged["tag"]
        df_4q_output["current_date_type"] = df_merged["date_type"]
        df_4q_output["current_presentation_role"] = df_merged["presentation_role"]

        for col in AXIS_COLS:
            df_4q_output["current_" + col] = df_merged[col]

        df_4q_output["start_current"] = df_merged["start_current"]
        df_4q_output["end_current"] = pd.NaT
        df_4q_output["current_period_value"] = df_merged["current_period_value_current"]
        df_4q_output["contextref_current"] = None

        df_4q_output["prior_start"] = df_merged["start_current"]
        df_4q_output["prior_end"] = pd.NaT
        df_4q_output["prior_period_value"] = df_merged["prior_period_value_current"]
        df_4q_output["contextref_prior"] = None
        df_4q_output["current_scale"] = df_merged.get("scale", df_merged.get("current_scale"))

        # === Standardize for downstream compatability
        df_4q_output = standardize_zip_output(df_4q_output)

        print(f"✅ Final 4Q output standardized: {len(df_4q_output)} rows")


    # In[120]:


    # === 4Q WORKFLOW =============================================
    # === IDENTIFY UNMATCHED ROWS AFTER EXACT MERGE ===

    if FOUR_Q_MODE:

        # Reuse the same merge key structure from exact match phase
        merge_key = ["tag"] + AXIS_COLS

        # FY rows that did NOT match in df_merged
        used_keys = set(df_merged["_key"])
        df_fy_unmatched = df_fy_matched[~df_fy_matched["_key"].isin(used_keys)]

        # YTD rows that did NOT match in df_merged
        df_ytd_unmatched = df_ytd_matched[~df_ytd_matched["_key"].isin(used_keys)]

        print(f"🔍 Unmatched FY rows: {len(df_fy_unmatched)}")
        print(f"🔍 Unmatched YTD rows: {len(df_ytd_unmatched)}")


    # In[121]:


    # === 4Q WORKFLOW =============================================
    # === FUZZY MATCH UNMATCHED ROWS ACROSS AXIS COLUMNS ===
    # Use fuzzy match as a fallback match

    from rapidfuzz import fuzz

    if FOUR_Q_MODE:

        fuzzy_matched_rows = []

        for i, row_fy in df_fy_unmatched.iterrows():
            for j, row_ytd in df_ytd_unmatched.iterrows():

                # === Match only if tag and start_current are exactly equal
                if row_fy["tag"] != row_ytd["tag"]:
                    continue

                # === Compare axis columns using partial_ratio
                passed = True
                for col in AXIS_COLS:
                    a = row_fy.get(col, "")
                    b = row_ytd.get(col, "")

                    if not (isinstance(a, str) and isinstance(b, str)):
                        passed = False
                        break

                    score = fuzz.partial_ratio(a, b)
                    if score < 80:
                        passed = False
                        break

                if passed:
                    # === Build fuzzy matched row (same format as df_merged)
                    fuzzy_matched_rows.append({
                        "tag": row_fy["tag"],
                        "date_type": row_fy["date_type"],
                        "start_current": row_fy["start_current"], #retained for reference, not used for merge
                        "end_current": pd.NaT,
                        "current_period_value_current": row_fy["current_period_value"],
                        "prior_period_value_current": row_fy["prior_period_value"],
                        "current_period_value_prior": row_ytd["current_period_value"],
                        "prior_period_value_prior": row_ytd["prior_period_value"],
                        "contextref_current": None,
                        "contextref_prior": None,
                        "presentation_role": row_fy["presentation_role"],
                        "scale": row_fy.get("scale"),
                        **{col: row_fy[col] for col in AXIS_COLS},
                        "_key": None  # optional: kept here as placeholder
                    })
                    break  # Stop after first match for each FY row

        # === Convert to DataFrame and append to df_merged
        df_fuzzy_merged = pd.DataFrame(fuzzy_matched_rows)

        print(f"✅ Added {len(df_fuzzy_merged)} fuzzy-matched rows to df_merged.")


    # In[122]:


    # === 4Q WORKFLOW =============================================
    # === COMBINE exact and fuzzy matches and finalize 4Q output

    if FOUR_Q_MODE:
        # Step 1: Combine exact + fuzzy
        df_merged = pd.concat([df_merged, df_fuzzy_merged], ignore_index=True)

        # Step 2: Calculate 4Q values
        df_merged["current_period_value"] = (
            df_merged["current_period_value_current"] - df_merged["current_period_value_prior"]
        )
        df_merged["prior_period_value"] = (
            df_merged["prior_period_value_current"] - df_merged["prior_period_value_prior"]
        )

        # Step 3: Standardize and clean for downstream use
        df_4q_output = standardize_zip_output(df_merged)


    # In[123]:


    # === 4Q WORKFLOW =============================================
    # === Audit Fuzzy Matches ===
    # Check the near-miss fuzzy matches (to make sure its correctly categorizing)

    from rapidfuzz import fuzz

    # === AUDIT: Fuzzy Near-Miss Logging ===
    if FOUR_Q_MODE:

        borderline_log = []

        for i, row_fy in df_fy_unmatched.iterrows():
            for j, row_ytd in df_ytd_unmatched.iterrows():

                if row_fy["tag"] != row_ytd["tag"]:
                    continue  # Only audit rows with same tag

                for col in AXIS_COLS:
                    a = row_fy.get(col, "")
                    b = row_ytd.get(col, "")
                    if not (isinstance(a, str) and isinstance(b, str)):
                        continue

                    score = fuzz.partial_ratio(a, b)
                    if 70 <= score < 80:
                        borderline_log.append({
                            "tag": row_fy["tag"],
                            "axis_column": col,
                            "FY_value": a,
                            "YTD_value": b,
                            "fuzzy_score": score
                        })

        # === Show audit result
        df_borderline_audit = pd.DataFrame(borderline_log)
        print(f"🔍 Borderline fuzzy matches (score 70–79): {len(df_borderline_audit)}")


    # In[124]:


    # === 4Q WORKFLOW =============================================
    # === Match Instants: Current Q vs Prior Q from 10-K ===

    if FOUR_Q_MODE:
        print("\n🏦 Matching instant facts (current_q vs prior_q) from 10-K...")

        # === Filter instants & current Q (full year in a 10-K) ===
        df_curr_inst = df_current_10k[df_current_10k["matched_category"] == "current_q"].copy()
        df_prior_inst = df_current_10k[df_current_10k["matched_category"] == "prior_q"].copy()
    
        # Filter for period_type = 'instant'
        df_curr_inst = df_curr_inst[df_curr_inst["period_type"] == "instant"].copy()
        df_prior_inst = df_prior_inst[df_prior_inst["period_type"] == "instant"].copy()

        # === Flatten presentation_role (some are lists)
        df_curr_inst["presentation_role"] = df_curr_inst["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )
        df_prior_inst["presentation_role"] = df_prior_inst["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )

        # === Fill axis values ===
        for col in AXIS_COLS:
            df_curr_inst[col] = df_curr_inst[col].fillna("__NONE__")
            df_prior_inst[col] = df_prior_inst[col].fillna("__NONE__")

        # === Match keys ===
        MATCH_COLS = ["tag", "presentation_role"] + AXIS_COLS
        MIN_KEYS = ["tag"]

        # === Trim
        df_curr_inst_trim = df_curr_inst[MATCH_COLS + ["value", "contextref", "date_type", "scale"]].copy()
        df_prior_inst_trim = df_prior_inst[MATCH_COLS + ["value", "contextref", "date_type", "scale"]].copy()

        # === Drop duplicates
        df_curr_inst_trim = df_curr_inst_trim.drop_duplicates()
        df_prior_inst_trim = df_prior_inst_trim.drop_duplicates()

        # === Adaptive match
        print("\n🧠 Matching current_q vs prior_q instants (balance sheet)...")
        match_keys = run_adaptive_match_keys(df_curr_inst_trim, df_prior_inst_trim, MATCH_COLS, MIN_KEYS)

        # === Zip match
        df_zip = zip_match_in_order(df_curr_inst_trim, df_prior_inst_trim, match_keys)
        df_instants = standardize_zip_output(df_zip)

        # === Preview & Log
        total_inst_curr = len(df_curr_inst_trim)
        matched_inst_rows = len(df_instants)
        match_rate_inst = matched_inst_rows / max(total_inst_curr, 1)

        print(f"✅ Instant match rate: {match_rate_inst * 100:.1f}%")
        log_metric("match_rate", {"instant": match_rate_inst})

    else:
        print("⚙️ Skipped: Not in 4Q mode.")


    # In[125]:


    # === FINALIZE 4Q COMBINED OUTPUT ==============================

    if FOUR_Q_MODE:
    
        # Combine flow and instant value matches
        df_final_combined = pd.concat([
            df_4q_output, # standardized fuzzy+exact
            df_instants # from balance sheet
        ], ignore_index=True)
    
        # === CLEANUP STEPS ============================================
    
        # Drop rows where both values are missing
        df_final_combined = df_final_combined.dropna(
            subset=["current_period_value", "prior_period_value"],
            how="all"
        )
        # Drop exact duplicate rows (same values, same tag, etc.)
        df_final_combined = df_final_combined.drop_duplicates()
    
        # === PREVIEW & LOG RESULTS ============================================

        total_combined_rows = len(df_final_combined)
        total_full_year_facts = len(df_fy_curr) + len(df_curr_inst)
        match_rate_final_4q = total_combined_rows / total_full_year_facts
        print(f"✅ Final 4Q match rate: {match_rate_final_4q * 100:.1f}%")

        print(f"• Flow-based rows: {len(df_4q_output)}")
        print(f"• Instant rows: {len(df_instants)}")
        print(f"• Final 4Q combined output created: {len(df_final_combined)} rows")

    # Only log final_match_rate in pure 4Q mode (not FY mode)
    if FOUR_Q_MODE and not FULL_YEAR_MODE:
        log_metric("final_match_rate", match_rate_final_4q)


    # In[126]:


    # === FULL YEAR WORKFLOW =============================================
    # === Match Full Year (current vs prior) from target 10-K ===

    if FOUR_Q_MODE and FULL_YEAR_MODE:
    
        print("\n📘 Matching full year facts (current_full_year vs prior_full_year) from 10-K...")

        # === Filter matched categories ===
        df_fy_curr = df_current_10k[df_current_10k["matched_category"] == "current_full_year"].copy()
        df_fy_prior = df_current_10k[df_current_10k["matched_category"] == "prior_full_year"].copy()

        # === Flatten presentation_role (some are lists)
        df_fy_curr["presentation_role"] = df_fy_curr["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )
        df_fy_prior["presentation_role"] = df_fy_prior["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )

        # === Fill axis values ===
        for col in AXIS_COLS:
            df_fy_curr[col] = df_fy_curr[col].fillna("__NONE__")
            df_fy_prior[col] = df_fy_prior[col].fillna("__NONE__")

        # === Match keys ===
        MATCH_COLS = ["tag", "presentation_role"] + AXIS_COLS
        MIN_KEYS = ["tag"]

        # === Trim
        df_fy_curr_trim = df_fy_curr[MATCH_COLS + ["start", "end", "value", "contextref", "date_type", "scale"]].copy()
        df_fy_prior_trim = df_fy_prior[MATCH_COLS + ["start", "end", "value", "contextref", "date_type", "scale"]].copy()

        # === Adaptive match
        match_keys = run_adaptive_match_keys(df_fy_curr_trim, df_fy_prior_trim, MATCH_COLS, MIN_KEYS)

        # === Zip match
        df_zip = zip_match_in_order(df_fy_curr_trim, df_fy_prior_trim, match_keys)
        df_full_year_matched = standardize_zip_output(df_zip)

        # === Preview results
        print(f"\n✅ Matched full-year rows (FY vs FY): {len(df_full_year_matched)}")
    
        total_fy_rows = len(df_fy_curr)
        matched_fy_rows = len(df_full_year_matched)
        match_rate_fy_final = matched_fy_rows / max(total_fy_rows, 1)
    
        log_metric("match_rate", {"fy_final": match_rate_fy_final})
        print(f"✅ FY final match rate: {match_rate_fy_final:.1%}")

    else:
        print("⚙️ Skipped: Not in 4Q mode.")


    # In[127]:


    # === FULL-YEAR WORKFLOW =======================================
    # === Combine FY flow and instant facts for export ============

    if FOUR_Q_MODE and FULL_YEAR_MODE:
        print("\n📦 Building Full Year Matched Output...")

        # Combine full-year flow values + balance sheet instants
        df_final_fy = pd.concat([
            df_full_year_matched,
            df_instants
        ], ignore_index=True)

    # === CLEANUP STEPS ============================================
    
        # Drop rows where both values are missing (sanity)
        df_final_fy = df_final_fy.dropna(subset=["current_period_value", "prior_period_value"], how="all")

        # Remove exact duplicates (row-level)
        df_final_fy = df_final_fy.drop_duplicates()

        # Optional: deduplicate by value pair only
        df_final_fy = df_final_fy.drop_duplicates(subset=["current_period_value", "prior_period_value"])

    # === PREVIEW & LOG RATES ==================================================

        print(f"✅ Final FY Combined {len(df_final_fy)} rows")

        total_full_year_facts = len(df_fy_curr) + len(df_curr_inst)
        final_fy_row_count = len(df_final_fy)
        match_rate_fy_final_combined = final_fy_row_count / max(total_full_year_facts, 1)

        print(f"✅ Final full-year match rate: {match_rate_fy_final_combined:.1%}")

        if FOUR_Q_MODE and FULL_YEAR_MODE:
            log_metric("final_match_rate", match_rate_fy_final_combined)
        
    else:
        print("⚙️ Skipped: Not in full-year mode.")

    if full_year_mode:
        return df_final_fy
    return df_final_combined


def match_quarterly(parsed: dict) -> pd.DataFrame:
    """Quarterly matcher (Q mode)."""
    FOUR_Q_MODE = False
    FULL_YEAR_MODE = False

    df_current = parsed.get("df_current")
    df_prior = parsed.get("df_prior")
    target_10q = parsed.get("target_10q")
    prior_10q = parsed.get("prior_10q")

    # === CONFIG ===

    if not FOUR_Q_MODE:
        # === Normal 10-Q build
    
        MATCH_COLS = ["tag", "date_type"] + AXIS_COLS

        # Step 1: Filter just current_q and prior_q
        df_curr_q = df_current[df_current["matched_category"].isin(["current_q", "current_ytd"])].copy()
        df_prior_q = df_current[df_current["matched_category"].isin(["prior_q", "prior_ytd"])].copy()

        for col in AXIS_COLS:
            df_curr_q[col] = df_curr_q[col].fillna("__NONE__")
            df_prior_q[col] = df_prior_q[col].fillna("__NONE__")
    
        # Step 2: Trim to needed columns
        df_curr_trim = df_curr_q[MATCH_COLS + ["start", "end", "value", "contextref", "presentation_role", "scale"]].copy()
        df_prior_trim = df_prior_q[MATCH_COLS + ["start", "end", "value", "contextref", "presentation_role", "scale"]].copy()

        # === DIAGNOSTIC: Count duplicate match groups
        curr_group_sizes = df_curr_trim.groupby(MATCH_COLS).size()
        prior_group_sizes = df_prior_trim.groupby(MATCH_COLS).size()
    
        print("🔁 Duplicate match groups:")
        print(f"  • In current: {(curr_group_sizes > 1).sum()} groups with >1 row")
        print(f"  • In prior  : {(prior_group_sizes > 1).sum()} groups with >1 row")

        # === Optional: Match key overlap diagnostic

        curr_keys = set(df_curr_trim.groupby(MATCH_COLS).groups.keys())
        prior_keys = set(df_prior_trim.groupby(MATCH_COLS).groups.keys())
        shared_keys = curr_keys & prior_keys

        shared_ratio = len(shared_keys) / max(len(curr_keys), 1)
        print(f"\n🔍 Matching on: {MATCH_COLS}")
        print(f"   • Current keys: {len(curr_keys)}")
        print(f"   • Prior keys  : {len(prior_keys)}")
        print(f"   • Shared keys : {len(shared_keys)} ({shared_ratio:.2%} of current)")

        # Step 3: Zip match
        df_zip = zip_match_in_order(df_curr_trim, df_prior_trim, MATCH_COLS)

        # Step 3b: Rename and align to FINAL_COLS structure
        df_final = standardize_zip_output(df_zip)
    
        # Step 4: Clean up nulls and tag blanks
        df_final = df_final[df_final["tag"].notna()]
        df_final = df_final[df_final["tag"].str.strip() != ""]

        # Step 5: Preview and Log
    
        match_rate_q = len(df_final) / max(len(df_curr_q), 1)
        log_metric("match_rate", {"quarterly": match_rate_q})
        print(f"\n✅ Match rate (quarterly facts): {match_rate_q:.1%}")
        print(f"📊 Final total matched rows: {len(df_final)}")

    else:
        # 4Q Special build
        pass


    # In[130]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Structured match: YTD and Instant facts using the Prior Year's Quarterly Filing ===

    if not FOUR_Q_MODE:
    
        # === STEP 0: Set column names and match criteria

        MATCH_COLS_YTD = ["tag", "date_type"] + AXIS_COLS
        MATCH_COLS_INSTANT = ["tag", "end", "date_type", "presentation_role"] + AXIS_COLS

        # === Minimum fallback keys
        MIN_MATCH_COLS_YTD = ["tag", "date_type"]
        MIN_MATCH_COLS_INSTANT = ["tag", "end", "date_type"]

        # === STEP 0: Standardize fill values
    
        for col in AXIS_COLS:
            df_current[col] = df_current[col].fillna("__NONE__")
            df_prior[col] = df_prior[col].fillna("__NONE__")
    
        # === STEP 0: Filter df_current for current_q and current_YTD
    
        df_curr_filtered = df_current[df_current["matched_category"].isin(["current_q", "current_ytd"])].copy()
        print(f"✅ Filtered df_current to {len(df_curr_filtered)} rows (current_q, current_ytd)")

        # Split into YTD and Instant for separate merges
    
        df_ytd = df_curr_filtered[df_curr_filtered["date_type"] == "YTD"].copy()
        df_instant = df_curr_filtered[df_curr_filtered["period_type"] == "instant"].copy()
        print(f"YTD rows: {len(df_ytd)}, Instant rows: {len(df_instant)}")
    
        # === STEP 2: Match YTD facts using prior y/y quarterly filing

        df_ytd_curr = df_ytd[df_ytd["matched_category"] == "current_ytd"].copy()
        df_ytd_prior = df_prior[df_prior["matched_category"] == "current_ytd"].copy()  # Prior Q stores prior YTD as "current_ytd"
    
        # Trim to match keys + data
        df_ytd_curr_trim = df_ytd_curr[MATCH_COLS_YTD + ["start", "end", "value", "contextref", "presentation_role", "scale"]].copy()
        df_ytd_prior_trim = df_ytd_prior[MATCH_COLS_YTD + ["start", "end", "value", "contextref", "presentation_role", "scale"]].copy()
    
        # === DIAGNOSTIC: Count duplicate match groups
        curr_ytd_group_sizes = df_ytd_curr_trim.groupby(MATCH_COLS_YTD).size()
        prior_ytd_group_sizes = df_ytd_prior_trim.groupby(MATCH_COLS_YTD).size()
    
        print(f"🔁 Duplicate YTD match groups:")
        print(f"  • In current: {(curr_ytd_group_sizes > 1).sum()} groups with >1 row")
        print(f"  • In prior  : {(prior_ytd_group_sizes > 1).sum()} groups with >1 row")

        df_ytd_curr_trim = df_ytd_curr_trim.drop_duplicates()
        df_ytd_prior_trim = df_ytd_prior_trim.drop_duplicates()

        print("\n🧠 YTD: Starting adaptive match key evaluation...")
    
        match_cols_ytd = run_adaptive_match_keys(df_ytd_curr_trim, df_ytd_prior_trim, MATCH_COLS_YTD, MIN_MATCH_COLS_YTD)
        df_ytd_merged = zip_match_in_order(df_ytd_curr_trim, df_ytd_prior_trim, match_cols_ytd)
        df_ytd_merged = standardize_zip_output(df_ytd_merged)

        # === STEP 3: Match Instant facts from prior y/y quarterly filing

        df_instant_curr = df_instant[df_instant["matched_category"] == "current_q"].copy()
        df_instant_prior = df_prior[
            (df_prior["matched_category"] == "current_q") &
            (df_prior["period_type"] == "instant")
        ].copy()

        #Date shifting logic
    
        # Step 1: Calculate the true delta between fiscal year-ends (in days)
        fye_curr = parse_date(target_10q["document_period_end"])
        fye_prior = parse_date(prior_10q["document_period_end"])
        year_delta = (fye_curr - fye_prior).days
    
        # Step 2: Add that exact day offset to prior instant 'end' values
        df_instant_prior["end"] = df_instant_prior["end"].apply(lambda x: x + pd.Timedelta(days=year_delta))

        #Turn presentation role data into string
        df_instant_curr["presentation_role"] = df_instant_curr["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )
        df_instant_prior["presentation_role"] = df_instant_prior["presentation_role"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else str(x)
        )

        #Create two match groups for sequential match
        df_instant_curr_trim = df_instant_curr[MATCH_COLS_INSTANT + ["value", "contextref", "scale"]].copy()
        df_instant_prior_trim = df_instant_prior[MATCH_COLS_INSTANT + ["value", "contextref", "scale"]].copy()

        # === DIAGNOSTIC: Count how many match groups have duplicate rows
    
        curr_group_sizes = df_instant_curr_trim.groupby(MATCH_COLS_INSTANT).size()
        prior_group_sizes = df_instant_prior_trim.groupby(MATCH_COLS_INSTANT).size()
    
        curr_dup_groups = (curr_group_sizes > 1).sum()
        prior_dup_groups = (prior_group_sizes > 1).sum()
    
        print(f"🔁 Duplicate match groups:")
        print(f"  • In current: {curr_dup_groups} groups with >1 row")
        print(f"  • In prior  : {prior_dup_groups} groups with >1 row")

        df_instant_curr_trim = df_instant_curr_trim.drop_duplicates()
        df_instant_prior_trim = df_instant_prior_trim.drop_duplicates()

        print("\n🧠 Instant: Starting adaptive match key evaluation...")

        match_cols_instant = run_adaptive_match_keys(df_instant_curr_trim, df_instant_prior_trim, MATCH_COLS_INSTANT, MIN_MATCH_COLS_INSTANT)
        df_instant_merged = zip_match_in_order(df_instant_curr_trim, df_instant_prior_trim, match_cols_instant)
        df_instant_merged = standardize_zip_output(df_instant_merged)
    
        print(f"✅ Instant facts matched (zip): {len(df_instant_merged)}")
    
        # === STEP 4: Append to df_final
        df_final = pd.concat(
            [df_final] + [f for f in [df_ytd_merged, df_instant_merged] if not f.empty],
            ignore_index=True
        )
    
        # === FINAL CLEANUP
        df_final = df_final.dropna(subset=["current_period_value", "prior_period_value"], how="any")
        df_final = df_final.drop_duplicates()
        print(f"✅ Final shape after dropping exact duplicates: {df_final.shape}")

        # === YTD Match Rate ===
        match_rate_ytd = len(df_ytd_merged) / max(len(df_ytd_curr_trim), 1)

        # === Instant Match Rate ===
        match_rate_instants = len(df_instant_merged) / max(len(df_instant_curr_trim), 1)

        # === Log both match rates
        log_metric("match_rate", {
            "ytd": match_rate_ytd,
            "instants": match_rate_instants
        })
    
        # === Optional: Preview ===
        print("\n✅ Filled missing prior_period_value from prior filing.")
    
        print(f"➕ YTD matches added: {len(df_ytd_merged)}")
        print(f"➕ Instant matches added: {len(df_instant_merged)}")

        print(f"✅ YTD match rate: {match_rate_ytd:.1%}")
        print(f"✅ Instant match rate: {match_rate_instants:.1%}")
    
        print("\n✅ Appended YTD and Instant fact matches to df_final")
        print(f"📊 Final total matched rows: {len(df_final)}")

        num_duplicates = df_final.duplicated(keep=False).sum()
        print(f"🔁 Total exact duplicate rows in df_final: {num_duplicates}")

    else:
        # === 4Q Special build
        pass


    # In[131]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Deeper Audit: Check if any tags in the current 10-Q were missed  ===

    if not FOUR_Q_MODE:

        # 1. All tags from df_current
        all_tags_in_current = set(df_current["tag"])
    
        # 2. Tags in the final export table
        tags_in_final = set(df_final["tag"])
    
        # 3. Find tags that exist in current but NOT in final
        missing_tags = all_tags_in_current - tags_in_final
    
        # === Output ===
        if missing_tags:
            print(f"\n⚡ {len(missing_tags)} tags exist in current 10-Q but were missing from df_final:")
            for tag in sorted(missing_tags):
                print(f"  - {tag}")

            # ✅ Build DataFrame before saving
            df_missing_tags = df_current[df_current["tag"].isin(missing_tags)].copy()

            # Keep relevant context columns
            df_missing_tags = df_missing_tags[[
                "tag", "value", "start", "end",
            ]].drop_duplicates()

            df_missing_tags["status_note"] = ""  # optional column for manual review            

        else:
            print("\n✅ No missing tags. All current_q tags are properly included in df_final.")

    else:
        # === 4Q Special build
        pass


    # In[132]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Select facts in df_current where tag is in missing tags to check values

    if not FOUR_Q_MODE:

        df_missing_facts = df_current[df_current["tag"].isin(missing_tags)]
    
        # === Optional: Show a quick preview
        print(f"\n✅ Found {len(df_missing_facts)} facts for missing tags.")

    else:
        # === 4Q Special build
        pass


    # In[133]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Fallback Match: Unmatched tags (YTD + Instant) with Current / Prior 10-Q ====================================

    if not FOUR_Q_MODE:

        fallback_keys = ["tag", "end", "date_type"] + AXIS_COLS  # Looser match (no presentation_role)

        # Filter to fallback tags
        df_curr_fallback = df_current[df_current["tag"].isin(missing_tags)].copy()
        df_prior_fallback = df_prior[df_prior["tag"].isin(missing_tags)].copy()

        # Filter to keep only matched_category current_q or current_ytd
        df_curr_fallback = df_curr_fallback[
            (df_curr_fallback["matched_category"] == "current_q") |
            (df_curr_fallback["matched_category"] == "current_ytd")
        ]
        df_prior_fallback = df_prior_fallback[
            (df_prior_fallback["matched_category"] == "current_q") |
            (df_prior_fallback["matched_category"] == "current_ytd")
        ]

        # Drop any entries without keys
        df_curr_fallback = df_curr_fallback.dropna(subset=fallback_keys)
        df_prior_fallback = df_prior_fallback.dropna(subset=fallback_keys)

        # Fill axis cols
        for col in AXIS_COLS: 
            df_curr_fallback[col] = df_curr_fallback[col].fillna("__NONE__")
            df_prior_fallback[col] = df_prior_fallback[col].fillna("__NONE__")

        # Shift dates ONLY for instant prior facts
    
        # === Step 1: Calculate the true day offset between fiscal year ends
        fye_curr = parse_date(target_10q["fiscal_year_end"])
        fye_prior = parse_date(prior_10q["fiscal_year_end"])
        year_delta = (fye_curr - fye_prior).days
    
        # === Step 2: Shift only instant-type rows in prior fallback
        mask = df_prior_fallback["period_type"] == "instant"
        df_prior_fallback.loc[mask, "end"] = df_prior_fallback.loc[mask, "end"] + pd.Timedelta(days=year_delta)

        # Run fallback match
        df_fallback_matches = zip_match_in_order(df_curr_fallback, df_prior_fallback, fallback_keys)

        print(f"\n✅ Fallback matches generated: {len(df_fallback_matches)} rows")
    
        # Clean final output
        df_fallback_clean = standardize_zip_output(df_fallback_matches)
        df_fallback_clean = df_fallback_clean.dropna(subset=["current_period_value", "prior_period_value"], how="all")
        df_fallback_clean = df_fallback_clean.drop_duplicates()
    
        # Log fallback match rate
        fallback_match_rate = len(df_fallback_clean) / max(len(df_curr_fallback), 1)
        log_metric("match_rate", {"fallback": fallback_match_rate})
        print(f"✅ Fallback match rate: {fallback_match_rate:.1%}")


    # In[134]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Collision Audit for Fallback Matches ===

    if not FOUR_Q_MODE:

        flagged_fallback_df = audit_value_collisions(df_fallback_clean)
        flagged_fallback_values = set(flagged_fallback_df["current_period_value"])

        df_fallback_clean["collision_flag"] = df_fallback_clean["current_period_value"].apply( 
            lambda x: 1 if x in flagged_fallback_values else 0
        )

        if not flagged_fallback_df.empty:
            print(f"⚠️ Fallback collision detected — {len(flagged_fallback_df)} flagged rows")

        else:
            print("✅ No collision flags in fallback output")


    # In[135]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Audit to check for duplicate prior year values in fallback and final output ====================================

    if not FOUR_Q_MODE:
    
        # Look for cases where the same prior value is matched to multiple current values
    
        # Ensure relevant columns are present and drop rows with missing values
        df_final_check = df_final[["tag", "current_period_value", "prior_period_value"]].dropna(subset=["prior_period_value"]).copy()
        df_fallback_check = df_fallback_clean[["tag", "current_period_value", "prior_period_value"]].dropna(subset=["prior_period_value"]).copy()
    
        # Convert prior values to numeric for safe matching
        df_final_check["prior_period_value"] = pd.to_numeric(df_final_check["prior_period_value"], errors="coerce")
        df_fallback_check["prior_period_value"] = pd.to_numeric(df_fallback_check["prior_period_value"], errors="coerce")
    
        # Identify overlaps by prior_period_value
        overlap = pd.merge(
            df_final_check,
            df_fallback_check,
            on="prior_period_value",
            suffixes=("_final", "_fallback")
        )
        
        if not overlap.empty:
            print(f"🔍 Found {len(overlap)} overlapping prior values between main and fallback outputs.")

        else:
            print("✅ No overlapping prior values found.")


    # In[136]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Audit to check for value mismatch between matched fallback and final outputs ====================================

    if not FOUR_Q_MODE:
        # Check for shared prior period values that were matched to different current period values in final vs. fallback datasets
    
        # Ensure aligned and numeric
        df_final_check = df_final[["tag", "current_period_value", "prior_period_value"]].dropna(subset=["prior_period_value"]).copy()
        df_fallback_check = df_fallback_clean[["tag", "current_period_value", "prior_period_value"]].dropna(subset=["prior_period_value"]).copy()
    
        df_final_check["prior_period_value"] = pd.to_numeric(df_final_check["prior_period_value"], errors="coerce")
        df_fallback_check["prior_period_value"] = pd.to_numeric(df_fallback_check["prior_period_value"], errors="coerce")
        df_final_check["current_period_value"] = pd.to_numeric(df_final_check["current_period_value"], errors="coerce")
        df_fallback_check["current_period_value"] = pd.to_numeric(df_fallback_check["current_period_value"], errors="coerce")
    
        # Merge to find overlapping prior values
        overlap = pd.merge(
            df_final_check,
            df_fallback_check,
            on="prior_period_value",
            suffixes=("_final", "_fallback")
        )
    
        # Find rows where current_period_value differs
        mismatches = overlap[
            overlap["current_period_value_final"] != overlap["current_period_value_fallback"]
        ]
    
        print(f"🔍 Found {len(mismatches)} mismatched current values for overlapping prior values.")


    # In[137]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Finalize fallback matches by removing overlapping prior values ============================

    if not FOUR_Q_MODE:
        
        # Step 1: Identify overlapping prior_period_values from earlier audit
        overlap_prior_values = set(overlap["prior_period_value"])
    
        # Step 2: Filter fallback matches to only keep non-duplicates
        df_fallback_unique = df_fallback_clean[
            ~df_fallback_clean["prior_period_value"].isin(overlap_prior_values)
        ].copy()
    
        print(f"✅ Result: {len(df_fallback_unique)} fallback matches added after removing {len(overlap_prior_values)} overlapping prior values.")


    # In[138]:


    # === NORMAL 10-Q WORKFLOW ====================================
    # === Append fallback matches to df_final before visual export ================

    if not FOUR_Q_MODE:
    
        # Ensure it's aligned to final structure
        df_fallback_finalized = df_fallback_unique.copy()
        df_fallback_finalized = standardize_zip_output(df_fallback_finalized)

        if not df_fallback_finalized.empty:
            # Append fallback to final
            df_final = pd.concat([df_final, df_fallback_finalized], ignore_index=True)
            df_final.reset_index(drop=True, inplace=True)
        
            print(f"✅ Final row count after fallback merge: {len(df_final)}")

        else:
            print("⚠️ df_fallback_finalized is empty — skipped appending to df_final.")

        # === Final match rate logging
        total_facts_current = len(df_current[df_current["matched_category"].isin(["current_q", "current_ytd"])])
        match_rate_final_quarter = len(df_final) / max(total_facts_current, 1)

        print(f"✅ Final quarterly match rate: {match_rate_final_quarter:.1%}")
        log_metric("final_match_rate", match_rate_final_quarter)

    return df_final


def match_4q(parsed: dict) -> pd.DataFrame:
    """4Q matcher (Q4 minus YTD path)."""
    return _match_4q_common(parsed, full_year_mode=False)


def match_fy(parsed: dict) -> pd.DataFrame:
    """Full-year matcher (FY path)."""
    return _match_4q_common(parsed, full_year_mode=True)


def match_filing(parsed: dict) -> pd.DataFrame:
    """Dispatch to the appropriate matcher and apply shared post-processing."""
    mode = parsed.get("mode")
    if mode == "Q":
        export_df = match_quarterly(parsed)
    elif mode == "4Q":
        export_df = match_4q(parsed)
    elif mode == "FY":
        export_df = match_fy(parsed)
    else:
        raise ValueError(f"Unsupported parsed mode: {mode}")

    export_df = _apply_collision_audit(export_df)
    negated_tags = parsed.get("negated_tags", set())
    export_df = apply_visual_signs(export_df.copy(), negated_tags)

    flipped_current = export_df.apply(
        lambda row: row["tag"] in negated_tags
        and pd.notna(row["current_period_value"])
        and row["visual_current_value"] != row["current_period_value"],
        axis=1,
    ).sum()
    flipped_prior = export_df.apply(
        lambda row: row["tag"] in negated_tags
        and pd.notna(row["prior_period_value"])
        and row["visual_prior_value"] != row["prior_period_value"],
        axis=1,
    ).sum()

    log_metric("sign_flip_count", {"current": int(flipped_current), "prior": int(flipped_prior)})
    print("🔁 Sign flip applied on:")
    print(f"   • Current period: {flipped_current} values")
    print(f"   • Prior period  : {flipped_prior} values")

    export_df = export_df.sort_values(by=["presentation_role", "tag"])
    return _finalize_export_df(export_df)
