# Claude Code Instructions for DocAnalyzer Project

## ⚠️ CRITICAL: Modification Restrictions

**YOU ARE ONLY AUTHORIZED TO MODIFY THE FOLLOWING:**
- Revenue extraction logic in `services/extraction_service.py`
- System prompt rules related to revenue and projection extraction
- Post-processing logic specifically for revenue validation

**DO NOT MODIFY:**
- Database schema or migration logic
- Flask routes or API endpoints
- Frontend code (HTML/CSS/JavaScript)
- OCR service logic
- Other financial field extraction (EBITDA, SG&A, CapEx, etc.) unless explicitly instructed
- Authentication, file upload, or storage paths
- Any configuration outside of revenue extraction

---

## 🎯 Current Task: Fix Revenue Extraction Accuracy

### Problem Statement
The AI extraction is producing incorrect revenue values that violate fundamental accounting rules:
- **Gross Profit > Net Revenue** (impossible - GP is derived FROM revenue)
- Revenue values are being confused with other P&L line items
- Historical revenue extraction is incomplete or inaccurate
- Projections are not properly based on historical growth patterns

### Root Cause Analysis
From `storage/extractions/3.json`:
```json
"net_revenue_hist": [50723, 45474, 47714],  // ❌ WRONG - too low
"gross_profit_hist": [54074, 40680, 36843], // ❌ GP > Revenue (impossible)
```

The AI is likely:
1. Reading segment/division revenue instead of total consolidated revenue
2. Confusing revenue with operating income or other smaller line items
3. Not following the table header row to identify ALL year columns

### Required Changes

#### 1. Enhanced System Prompt - Revenue Extraction Rules

**Location:** `services/extraction_service.py` - `SYSTEM_PROMPT` variable (around line 23)

**Add these rules to the REVENUE section:**

```
══════════════════════════════════════
REVENUE EXTRACTION — ENHANCED RULES
══════════════════════════════════════

CRITICAL: Revenue is the LARGEST number in the Income Statement.
Revenue MUST be greater than Gross Profit (GP is revenue minus COGS).
If your extracted Revenue < Gross Profit, YOU HAVE THE WRONG ROW.

STEP-BY-STEP REVENUE EXTRACTION:
1. Locate the CONSOLIDATED Income Statement (NOT segment breakdowns)
2. Find the FIRST line of the P&L — this is ALWAYS revenue
3. Common labels: "Net Revenue", "Total Revenue", "Net Sales", "Revenue"
4. Read the table header row to identify ALL year columns (typically 3-6 years)
5. For EACH year column, extract the revenue value from that FIRST row
6. Return an array with one value per year column

VALIDATION CHECKS (apply BEFORE returning):
- Revenue must be the LARGEST value in each column of the P&L
- Revenue MUST be > Gross Profit for every year
- Revenue MUST be > SG&A for every year
- Revenue MUST be > EBITDA for every year
- Typical revenue range for PE targets: $30M-$500M ($30,000-$500,000 in $000s)
- If any validation fails: re-read the table and find the ACTUAL top-line revenue

HISTORICAL REVENUE (net_revenue_hist):
- Extract from the main P&L table first row
- Must have exactly 3 values (one per historical year column)
- NEVER return [value, null, null] when 3 columns exist
- Cross-check: Calculate implied COGS = Revenue - Gross Profit
  - COGS should be 20-60% of revenue for manufacturing companies
  - If COGS < 0 or > revenue, you extracted the wrong revenue row

PROJECTION REVENUE (net_revenue_proj):
- Extract from the projection table/model (separate section from historical)
- Must have exactly 5 values (pad with null if fewer years shown)
- Calculate implied growth rates year-over-year
- Typical projection growth: -10% to +30% per year
- If growth rate > 50% or < -30%, verify the extracted values

REVENUE PROJECTION QUALITY CHECKS:
- Compare Year 1 projection to most recent historical year
- Growth should be reasonable based on historical trends
- If historical revenue is declining, projections should reflect realistic turnaround
- Do NOT accept flat/identical projections (e.g., [100, 100, 100, 100, 100])
  unless the document explicitly shows flat revenue assumptions
```

#### 2. Add Revenue-Specific Validation Function

**Location:** `services/extraction_service.py` - Add new function after `check_revenue_completeness()`

```python
def validate_revenue_accuracy(data):
    """
    Validates that revenue values make accounting sense.
    Revenue must ALWAYS be greater than Gross Profit, SG&A, and EBITDA.
    """
    f = data.get("financials", {})
    corrections = []

    rev_h = f.get("net_revenue_hist", [None] * 3)
    gp_h = f.get("gross_profit_hist", [None] * 3)
    sga_h = f.get("sga_hist", [None] * 3)
    ebitda_h = f.get("adj_ebitda_hist", [None] * 3)

    # Historical revenue validation
    for i in range(min(3, len(rev_h))):
        if rev_h[i] is None:
            continue

        # Check 1: Revenue > Gross Profit
        if gp_h[i] is not None and rev_h[i] < gp_h[i]:
            corrections.append(
                f"REVENUE ERROR hist[{i}]: Revenue({rev_h[i]}) < Gross Profit({gp_h[i]}) "
                f"— IMPOSSIBLE. This violates basic accounting. Revenue is the wrong row."
            )
            # Attempt auto-fix: if GP looks reasonable as revenue, swap them
            if gp_h[i] > rev_h[i] * 1.5:  # GP is significantly larger
                old_rev = rev_h[i]
                rev_h[i] = gp_h[i]
                gp_h[i] = old_rev
                corrections.append(
                    f"AUTO-FIX hist[{i}]: Swapped Revenue and Gross Profit. "
                    f"New Revenue={rev_h[i]}, New GP={gp_h[i]}"
                )

        # Check 2: Revenue > SG&A
        if sga_h[i] is not None and rev_h[i] < sga_h[i]:
            corrections.append(
                f"REVENUE ERROR hist[{i}]: Revenue({rev_h[i]}) < SG&A({sga_h[i]}) "
                f"— Revenue is likely from a segment table, not consolidated P&L."
            )

        # Check 3: Revenue > EBITDA
        if ebitda_h[i] is not None and rev_h[i] < ebitda_h[i]:
            corrections.append(
                f"REVENUE ERROR hist[{i}]: Revenue({rev_h[i]}) < EBITDA({ebitda_h[i]}) "
                f"— Revenue extracted from wrong table row."
            )

    # Projection revenue validation
    rev_p = f.get("net_revenue_proj", [None] * 5)
    for i in range(min(5, len(rev_p))):
        if rev_p[i] is not None and i > 0 and rev_p[i - 1] is not None:
            growth = (rev_p[i] - rev_p[i - 1]) / rev_p[i - 1]
            if growth > 0.50:  # > 50% growth
                corrections.append(
                    f"REVENUE WARNING proj[{i}]: Growth of {growth:.1%} seems high. "
                    f"Verify projection values."
                )
            elif growth < -0.30:  # > 30% decline
                corrections.append(
                    f"REVENUE WARNING proj[{i}]: Decline of {growth:.1%} seems steep. "
                    f"Verify projection values."
                )

    f["net_revenue_hist"] = rev_h
    f["gross_profit_hist"] = gp_h
    data["financials"] = f

    if corrections:
        print(f"[REVENUE VALIDATION] {len(corrections)} issues found:")
        for c in corrections:
            print(f"  >> {c}")
        data.setdefault("_revenue_corrections", []).extend(corrections)

    return data
```

#### 3. Update Extraction Pipeline

**Location:** `services/extraction_service.py` - Function `extract_document_financials()` (line 1361)

**Modify the pipeline sequence (around line 1389-1402):**

```python
# Current pipeline:
extraction_data = check_and_fix_row_swaps(extraction_data)
extraction_data = check_revenue_completeness(extraction_data)
extraction_data = post_process_extraction(extraction_data)
extraction_data = validate_and_correct(extraction_data)
extraction_data = derive_missing_values(extraction_data)

# NEW pipeline - add revenue validation as FIRST step:
extraction_data = validate_revenue_accuracy(extraction_data)  # ← ADD THIS FIRST
extraction_data = check_and_fix_row_swaps(extraction_data)
extraction_data = check_revenue_completeness(extraction_data)
extraction_data = post_process_extraction(extraction_data)
extraction_data = validate_and_correct(extraction_data)
extraction_data = derive_missing_values(extraction_data)
```

#### 4. Enhanced Projection Logic

**Add to SYSTEM_PROMPT after the revenue section:**

```
REVENUE PROJECTION METHODOLOGY:
When extracting projections, look for:
1. Management case / base case projections (primary source)
2. Banker projections / financial advisor forecasts (secondary)
3. If multiple scenarios exist (bull/base/bear), use BASE case

Revenue projection quality indicators:
- Year 1 projection should align with latest trends or guidance
- Growth rates should show a pattern (acceleration, steady, deceleration)
- Terminal years (Year 4-5) often flatten or use lower growth rates
- If projections are missing: DO NOT fabricate — return null

Cross-reference projections with:
- Historical CAGR (Compound Annual Growth Rate)
- Management commentary on growth initiatives
- Market outlook discussions in the CIM
```

---

## 🧪 Testing After Changes

1. **Re-upload document ID 3** (Polytek Development Corp) and verify:
   - Revenue > Gross Profit for all years
   - No "impossible" corrections in `_corrections_applied`
   - Gross margin (GP/Revenue) is between 20-80%

2. **Check extraction JSON** for these fields:
   ```json
   "net_revenue_hist": [X, Y, Z],  // Should be LARGEST numbers
   "gross_profit_hist": [A, B, C], // Must be < corresponding revenue
   "_revenue_corrections": []       // Should be empty or minimal
   ```

3. **Compare against OCR text** manually:
   - Find the main P&L table in `storage/ocr/processed/{doc_id}.txt`
   - Verify extracted revenue matches the TOP LINE of that table

---

## 📊 Success Criteria

✅ Revenue values are the largest numbers in each year's P&L column
✅ Gross Profit < Revenue for ALL years (historical + projections)
✅ Revenue extraction confidence = "high" in field_level
✅ No revenue-related swaps/corrections in pipeline
✅ Projection growth rates are reasonable (-30% to +50%)
✅ All 3 historical years extracted (no [value, null, null] arrays)
✅ All 5 projection years extracted or properly padded with nulls

---

## 🔒 Reminder: Scope Boundaries

**ONLY modify code related to revenue extraction validation and prompt rules.**

If you discover issues with other fields (EBITDA, CapEx, SG&A), **document them** but do NOT fix them unless explicitly instructed by the user.

When in doubt: **ask the user before making changes outside revenue logic.**
