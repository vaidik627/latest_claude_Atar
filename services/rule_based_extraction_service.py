"""
Rule-Based Financial Data Extraction Service

This module extracts structured financial data from OCR text using deterministic
pattern matching and table parsing. Replaces LLM-based extraction to eliminate
hallucinations and ensure consistent, accurate results.

Architecture: OCR Text → Table Detection → Pattern Matching → Validation → JSON Output
"""

import os
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from fuzzywuzzy import fuzz
from database import update_document_extraction

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EXTRACTIONS_FOLDER = os.path.join(BASE_DIR, 'storage/extractions')
os.makedirs(EXTRACTIONS_FOLDER, exist_ok=True)

# ============================================================================
# PATTERN DEFINITIONS
# ============================================================================

# Revenue patterns (first line of P&L, largest number)
REVENUE_PATTERNS = [
    r"(?:Net\s+)?Revenue",
    r"Total\s+Revenue",
    r"Revenue,?\s+net",
    r"Net\s+Sales",
    r"Sales",
]

# Gross Profit patterns
GROSS_PROFIT_PATTERNS = [
    r"Gross\s+Profit",
    r"Gross\s+Margin",
    r"GP",
]

# COGS patterns
COGS_PATTERNS = [
    r"Cost\s+of\s+(?:Goods\s+Sold|Sales|Revenue)",
    r"COGS",
    r"Direct\s+Costs",
]

# SG&A patterns
SGA_PATTERNS = [
    r"(?:SG&A|SGA)",
    r"Selling,?\s+General\s+(?:and|&)\s+Administrative",
    r"Operating\s+Expenses",
]

# EBITDA patterns (must be LARGER than Operating Income)
EBITDA_PATTERNS = [
    r"Adjusted\s+EBITDA",
    r"Adj\.?\s+EBITDA",
    r"EBITDA\s+\(Adjusted\)",
    r"Normalized\s+EBITDA",
    r"EBITDA",
]

# Operating Income patterns (SMALLER than EBITDA)
OPERATING_INCOME_PATTERNS = [
    r"Operating\s+Income",
    r"Income\s+from\s+Operations",
    r"EBIT(?!\s*DA)",  # EBIT but not EBITDA
]

# Depreciation & Amortization patterns
DA_PATTERNS = [
    r"Depreciation\s+(?:and|&)\s+Amortization",
    r"D&A",
    r"DA",
    r"Amortization",
]

# CapEx patterns
CAPEX_PATTERNS = [
    r"Capital\s+Expenditures?",
    r"CapEx",
    r"Cap\s+Ex",
    r"PP&E\s+Purchases",
]

# Adjustments/Add-backs patterns
ADJUSTMENTS_PATTERNS = [
    r"(?:EBITDA\s+)?Adjustments",
    r"Add[\s\-]?backs",
    r"One[\s\-]?time\s+(?:Items|Expenses)",
    r"Non[\s\-]?recurring",
]

# Management Fees patterns
MGMT_FEES_PATTERNS = [
    r"Management\s+Fees?",
    r"Mgmt\.?\s+Fees?",
]

# Entry Multiple / Purchase Price patterns
ENTRY_MULTIPLE_PATTERNS = [
    r"Entry\s+Multiple",
    r"Purchase\s+Price\s+Multiple",
    r"Valuation\s+Multiple",
]

PURCHASE_PRICE_PATTERNS = [
    r"Purchase\s+Price",
    r"Transaction\s+Value",
    r"Enterprise\s+Value",
]

# Collateral patterns
AR_PATTERNS = [r"Accounts?\s+Receivable", r"A/?R", r"Trade\s+Receivables?"]
INVENTORY_PATTERNS = [r"Inventory", r"Inventories"]
EQUIPMENT_PATTERNS = [r"Equipment", r"Machinery", r"PP&E"]

# Rates patterns
ABL_RATE_PATTERNS = [r"ABL\s+Rate", r"Revolver\s+Rate", r"Line\s+Rate"]
TERM_RATE_PATTERNS = [r"Term\s+(?:Loan\s+)?Rate", r"TL\s+Rate"]

# Year patterns
HISTORICAL_YEAR_PATTERNS = [
    r"(?:Dec|FYE?|Q4)[\s\-]?(\d{2,4})A",  # Dec-22A, FY23A, Q4-2022A
    r"(\d{4})A",  # 2022A
]

PROJECTION_YEAR_PATTERNS = [
    r"(?:Dec|FYE?|Q4)[\s\-]?(\d{2,4})[FEP]",  # Dec-26F, FY27E, Q4-2028P
    r"(\d{4})[FEP]",  # 2026F
]

# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_document_financials(doc_id: int, ocr_text_path: str) -> None:
    """
    Main extraction pipeline - replaces LLM-based extraction.

    Args:
        doc_id: Document ID in database
        ocr_text_path: Relative path to OCR processed text file
    """
    try:
        print(f"[RULE-BASED EXTRACTION] doc_id={doc_id}: Started")

        # Update status to processing
        update_document_extraction(
            doc_id,
            extraction_status='processing'
        )

        # 1. Load OCR text
        full_ocr_path = os.path.join(BASE_DIR, ocr_text_path)
        with open(full_ocr_path, 'r', encoding='utf-8') as f:
            ocr_text = f.read()

        print(f"[EXTRACTION] Loaded OCR text: {len(ocr_text)} chars")

        # 2. Parse document structure
        tables = extract_all_tables(ocr_text)
        print(f"[EXTRACTION] Found {len(tables)} tables")

        # 3. Identify key tables
        income_statement = find_table_by_type(tables, 'income_statement')
        cash_flow = find_table_by_type(tables, 'cash_flow')
        balance_sheet = find_table_by_type(tables, 'balance_sheet')
        projections = find_table_by_type(tables, 'projections')

        # 4. Extract metadata
        company_name = extract_company_name(ocr_text)
        industry = extract_industry(ocr_text)

        # 5. Extract year columns
        historical_years = extract_year_columns(income_statement, is_historical=True) if income_statement is not None else []
        projection_years = extract_year_columns(projections, is_historical=False) if projections is not None else []

        print(f"[EXTRACTION] Historical years: {historical_years}")
        print(f"[EXTRACTION] Projection years: {projection_years}")

        # 6. Extract financial data
        financials = extract_financials(
            income_statement=income_statement,
            cash_flow=cash_flow,
            projections=projections,
            num_hist_years=len(historical_years),
            num_proj_years=max(5, len(projection_years))  # Always 5 projection years
        )

        # 7. Extract deal metrics
        deal = extract_deal_metrics(ocr_text, tables, financials)

        # 8. Extract collateral
        collateral = extract_collateral(balance_sheet, tables)

        # 9. Extract rates
        rates = extract_rates(ocr_text)

        # 10. Extract fees
        fees = extract_fees(ocr_text)

        # 11. Extract qualitative data
        qualitative = extract_qualitative(ocr_text)

        # 12. Build extraction result
        extraction_data = {
            "company_name": company_name,
            "industry": industry,
            "geography": extract_geography(ocr_text),
            "transaction_date": None,  # Usually not in CIM
            "historical_years": historical_years,
            "projection_years": projection_years[:5],  # Limit to 5
            "financials": financials,
            "deal": deal,
            "collateral": collateral,
            "rates": rates,
            "fees": fees,
            "qualitative": qualitative,
            "confidence": calculate_confidence(extraction_data=None),  # Placeholder
            "_corrections_applied": [],
            "_extraction_method": "rule_based",
            "_extraction_timestamp": datetime.now(timezone.utc).isoformat()
        }

        # 13. Validate extractions
        extraction_data, validation_warnings = validate_extraction(extraction_data)
        extraction_data["_validation_warnings"] = validation_warnings

        # 14. Apply business rules
        extraction_data = apply_business_rules(extraction_data)

        # 15. Calculate confidence scores
        extraction_data["confidence"] = calculate_confidence(extraction_data)

        # 16. Save extraction JSON
        extraction_path = os.path.join(EXTRACTIONS_FOLDER, f'{doc_id}.json')
        with open(extraction_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)

        print(f"[EXTRACTION] Saved to {extraction_path}")

        # 17. Update database
        update_document_extraction(
            doc_id,
            extraction_status='completed',
            extraction_completed_at=datetime.now(timezone.utc).isoformat(),
            extraction_path=f'storage/extractions/{doc_id}.json',
            company_name=company_name,
            confidence_score=extraction_data["confidence"].get("overall_confidence", 0) / 100.0,
            ebitda_ltm=deal.get("ebitda_ltm"),
            revenue_ltm=deal.get("revenue_ltm"),
            entry_multiple=deal.get("entry_multiple"),
            purchase_price=deal.get("purchase_price_calculated"),
        )

        print(f"[EXTRACTION] doc_id={doc_id}: Completed successfully")

    except Exception as e:
        print(f"[EXTRACTION ERROR] doc_id={doc_id}: {e}")
        import traceback
        traceback.print_exc()

        update_document_extraction(
            doc_id,
            extraction_status='failed',
            extraction_error=str(e)
        )

# ============================================================================
# TABLE DETECTION AND PARSING
# ============================================================================

def extract_all_tables(ocr_text: str) -> List[pd.DataFrame]:
    """
    Extract all tables from OCR text.
    OCR service formats tables as markdown pipes (|).
    """
    tables = []

    # Pattern to match markdown-style tables
    # Format: | col1 | col2 |\n| ---- | ---- |\n| val1 | val2 |
    table_pattern = r'\|([^\n]+)\|\n\|([\s\-|]+)\|\n((?:\|[^\n]+\|\n?)+)'

    for match in re.finditer(table_pattern, ocr_text, re.MULTILINE):
        try:
            table_text = match.group(0)
            df = parse_markdown_table(table_text)
            if df is not None and not df.empty:
                tables.append(df)
        except Exception as e:
            print(f"[TABLE PARSE WARNING] Failed to parse table: {e}")
            continue

    return tables

def parse_markdown_table(table_text: str) -> Optional[pd.DataFrame]:
    """Parse a markdown-formatted table into a pandas DataFrame."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip()]

    if len(lines) < 3:  # Need at least header, separator, and one data row
        return None

    # Extract header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]

    # Skip separator line (line[1])

    # Extract data rows
    data_rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if len(cells) == len(headers):
            data_rows.append(cells)

    if not data_rows:
        return None

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=headers)

    # Convert numeric columns
    for col in df.columns[1:]:  # Skip first column (labels)
        df[col] = df[col].apply(parse_numeric_value)

    return df

def parse_numeric_value(text: str) -> Optional[float]:
    """
    Parse numeric value from text, handling:
    - Parentheses for negative: (123) → -123
    - Thousands separators: 1,234.56
    - Dollar signs: $1,234
    - Percentages: 15.5% → 0.155
    - Units: 123M, 45.6K
    """
    if not text or text in ['--', '-', 'N/A', 'n/a', '']:
        return None

    text = text.strip()

    # Handle percentages
    if '%' in text:
        try:
            num = float(text.replace('%', '').replace(',', '').strip())
            return num / 100.0
        except:
            return None

    # Handle parentheses (negative numbers)
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1].strip()

    # Remove currency symbols and commas
    text = text.replace('$', '').replace(',', '').strip()

    # Handle units (M = millions, K = thousands)
    multiplier = 1.0
    if text.endswith('M') or text.endswith('m'):
        multiplier = 1000.0  # Already in thousands after removing M
        text = text[:-1].strip()
    elif text.endswith('K') or text.endswith('k'):
        multiplier = 1.0
        text = text[:-1].strip()

    try:
        num = float(text) * multiplier
        return -num if is_negative else num
    except:
        return None

def find_table_by_type(tables: List[pd.DataFrame], table_type: str) -> Optional[pd.DataFrame]:
    """
    Identify table type (income statement, cash flow, balance sheet, projections)
    based on row labels and structure.
    """
    for table in tables:
        if table.empty or len(table.columns) < 2:
            continue

        # Get first column (labels)
        labels = table.iloc[:, 0].astype(str).str.lower()

        if table_type == 'income_statement':
            # Look for P&L keywords
            if any('revenue' in l or 'sales' in l for l in labels):
                if any('gross profit' in l or 'ebitda' in l for l in labels):
                    return table

        elif table_type == 'cash_flow':
            # Look for cash flow keywords
            if any('operating activit' in l or 'net income' in l for l in labels):
                if any('depreciation' in l or 'amortization' in l for l in labels):
                    return table

        elif table_type == 'balance_sheet':
            # Look for balance sheet keywords
            if any('asset' in l or 'liabilit' in l for l in labels):
                if any('receivable' in l or 'inventory' in l for l in labels):
                    return table

        elif table_type == 'projections':
            # Look for projection year markers in columns
            for col in table.columns[1:]:
                if re.search(r'\d{2,4}[FEP]', str(col)):
                    # Check if this table also has revenue/EBITDA
                    if any('revenue' in l or 'ebitda' in l for l in labels):
                        return table

    return None

# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_company_name(ocr_text: str) -> Optional[str]:
    """Extract company name from document."""
    # Look for patterns near the beginning of the document
    first_500_chars = ocr_text[:500]

    # Pattern: Look for capitalized company names before "Confidential"
    match = re.search(r'^([A-Z][A-Za-z\s&,\.]+(?:Corp|Inc|LLC|Ltd|Company|Group)\.?)', first_500_chars, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Pattern: Look for company name in header
    match = re.search(r'===\s*PAGE\s*\d+\s*===\s*\n\s*([A-Z][A-Za-z\s&,\.]+)', ocr_text)
    if match:
        name = match.group(1).strip()
        if len(name) < 50 and 'confidential' not in name.lower():
            return name

    return None

def extract_industry(ocr_text: str) -> Optional[str]:
    """Extract industry/sector from document."""
    # Look for explicit industry mentions
    patterns = [
        r'Industry:\s*([A-Za-z\s&,]+)',
        r'Sector:\s*([A-Za-z\s&,]+)',
        r'Business:\s*([A-Za-z\s&,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None

def extract_geography(ocr_text: str) -> Optional[str]:
    """Extract primary geography/location."""
    # Look for location mentions
    patterns = [
        r'(?:Headquarters|Based in|Located in):\s*([A-Za-z\s,]+)',
        r'Geography:\s*([A-Za-z\s,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Default heuristic: look for US states or countries
    if re.search(r'\b(?:United States|USA|U\.S\.)\b', ocr_text):
        return "United States"

    return None

def extract_year_columns(table: Optional[pd.DataFrame], is_historical: bool) -> List[str]:
    """Extract year column headers from table."""
    if table is None or table.empty:
        return []

    years = []
    pattern = HISTORICAL_YEAR_PATTERNS if is_historical else PROJECTION_YEAR_PATTERNS

    for col in table.columns[1:]:  # Skip first column (labels)
        col_str = str(col)
        for p in pattern:
            if re.search(p, col_str, re.IGNORECASE):
                years.append(col_str)
                break

    return years

# ============================================================================
# FINANCIAL DATA EXTRACTION
# ============================================================================

def extract_financials(income_statement: Optional[pd.DataFrame],
                       cash_flow: Optional[pd.DataFrame],
                       projections: Optional[pd.DataFrame],
                       num_hist_years: int,
                       num_proj_years: int) -> Dict[str, Any]:
    """Extract all financial line items."""

    financials = {}

    # Historical financials (from income statement)
    if income_statement is not None and num_hist_years > 0:
        financials['net_revenue_hist'] = extract_by_pattern(income_statement, REVENUE_PATTERNS, num_hist_years)
        financials['gross_profit_hist'] = extract_by_pattern(income_statement, GROSS_PROFIT_PATTERNS, num_hist_years)
        financials['sga_hist'] = extract_by_pattern(income_statement, SGA_PATTERNS, num_hist_years)
        financials['adj_ebitda_hist'] = extract_by_pattern(income_statement, EBITDA_PATTERNS, num_hist_years)
        financials['adjustments_hist'] = extract_by_pattern(income_statement, ADJUSTMENTS_PATTERNS, num_hist_years)

        # D&A from cash flow (more reliable) or income statement
        if cash_flow is not None:
            financials['depreciation_hist'] = extract_by_pattern(cash_flow, DA_PATTERNS, num_hist_years)
        else:
            financials['depreciation_hist'] = extract_by_pattern(income_statement, DA_PATTERNS, num_hist_years)

        # CapEx from cash flow
        if cash_flow is not None:
            financials['capex_hist'] = extract_by_pattern(cash_flow, CAPEX_PATTERNS, num_hist_years)
            # Ensure CapEx is negative
            if financials['capex_hist']:
                financials['capex_hist'] = [
                    -abs(v) if v is not None else None
                    for v in financials['capex_hist']
                ]
        else:
            financials['capex_hist'] = [None] * num_hist_years

        # Calculate margins
        financials['ebitda_margin_hist'] = calculate_margins(
            financials.get('adj_ebitda_hist'),
            financials.get('net_revenue_hist')
        )
        financials['gm_pct_hist'] = calculate_margins(
            financials.get('gross_profit_hist'),
            financials.get('net_revenue_hist')
        )
        financials['revenue_growth_hist'] = calculate_growth_rates(
            financials.get('net_revenue_hist')
        )
    else:
        # Return nulls
        for key in ['net_revenue_hist', 'gross_profit_hist', 'sga_hist', 'adj_ebitda_hist',
                    'adjustments_hist', 'depreciation_hist', 'capex_hist',
                    'ebitda_margin_hist', 'gm_pct_hist', 'revenue_growth_hist']:
            financials[key] = [None] * num_hist_years

    # Projection financials
    if projections is not None and num_proj_years > 0:
        financials['net_revenue_proj'] = extract_by_pattern(projections, REVENUE_PATTERNS, num_proj_years)
        financials['gross_profit_proj'] = extract_by_pattern(projections, GROSS_PROFIT_PATTERNS, num_proj_years)
        financials['sga_proj'] = extract_by_pattern(projections, SGA_PATTERNS, num_proj_years)
        financials['adj_ebitda_proj'] = extract_by_pattern(projections, EBITDA_PATTERNS, num_proj_years)
        financials['adjustments_proj'] = extract_by_pattern(projections, ADJUSTMENTS_PATTERNS, num_proj_years)
        financials['depreciation_proj'] = extract_by_pattern(projections, DA_PATTERNS, num_proj_years)
        financials['capex_proj'] = extract_by_pattern(projections, CAPEX_PATTERNS, num_proj_years)

        # Ensure CapEx is negative
        if financials['capex_proj']:
            financials['capex_proj'] = [
                -abs(v) if v is not None else None
                for v in financials['capex_proj']
            ]

        financials['mgmt_fees_proj'] = extract_by_pattern(projections, MGMT_FEES_PATTERNS, num_proj_years)
        # Ensure mgmt fees are negative
        if financials['mgmt_fees_proj']:
            financials['mgmt_fees_proj'] = [
                -abs(v) if v is not None and v != 0 else (0 if v == 0 else None)
                for v in financials['mgmt_fees_proj']
            ]

        # Calculate projection margins
        financials['ebitda_margin_proj'] = calculate_margins(
            financials.get('adj_ebitda_proj'),
            financials.get('net_revenue_proj')
        )
        financials['gm_pct_proj'] = calculate_margins(
            financials.get('gross_profit_proj'),
            financials.get('net_revenue_proj')
        )
    else:
        # Return nulls
        for key in ['net_revenue_proj', 'gross_profit_proj', 'sga_proj', 'adj_ebitda_proj',
                    'adjustments_proj', 'depreciation_proj', 'capex_proj', 'mgmt_fees_proj',
                    'ebitda_margin_proj', 'gm_pct_proj']:
            financials[key] = [None] * num_proj_years

    financials['other_income_hist'] = [None] * num_hist_years  # Rarely in CIMs

    return financials

def extract_by_pattern(table: pd.DataFrame, patterns: List[str], num_years: int) -> List[Optional[float]]:
    """
    Extract values from table row matching any of the given patterns.
    Returns list of values for all year columns.
    """
    if table is None or table.empty:
        return [None] * num_years

    # Get label column
    labels = table.iloc[:, 0].astype(str)

    # Find matching row
    for idx, label in enumerate(labels):
        if matches_any_pattern(label, patterns):
            # Extract values from year columns (skip first column which is labels)
            values = table.iloc[idx, 1:1+num_years].tolist()
            # Pad with None if needed
            while len(values) < num_years:
                values.append(None)
            return values[:num_years]

    return [None] * num_years

def matches_any_pattern(text: str, patterns: List[str], fuzzy_threshold: int = 85) -> bool:
    """Check if text matches any of the given regex patterns (with fuzzy matching fallback)."""
    text_lower = text.lower()

    # First try exact regex matching
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Fallback to fuzzy matching for typos/variations
    for pattern in patterns:
        # Convert regex pattern to plain text for fuzzy matching
        pattern_text = pattern.replace(r'\s+', ' ').replace(r'(?:', '').replace(')', '').replace('\\', '')
        pattern_text = re.sub(r'[^a-zA-Z\s]', '', pattern_text).strip().lower()

        if fuzz.partial_ratio(text_lower, pattern_text) >= fuzzy_threshold:
            return True

    return False

def calculate_margins(numerator: Optional[List[float]], denominator: Optional[List[float]]) -> List[Optional[float]]:
    """Calculate margin percentages."""
    if not numerator or not denominator:
        return [None] * (len(numerator) if numerator else 0)

    margins = []
    for num, den in zip(numerator, denominator):
        if num is not None and den is not None and den != 0:
            margins.append(round(num / den, 4))
        else:
            margins.append(None)

    return margins

def calculate_growth_rates(values: Optional[List[float]]) -> List[Optional[float]]:
    """Calculate year-over-year growth rates."""
    if not values or len(values) < 2:
        return [None] * (len(values) if values else 0)

    growth = [None]  # First year has no prior year
    for i in range(1, len(values)):
        if values[i] is not None and values[i-1] is not None and values[i-1] != 0:
            growth.append(round((values[i] - values[i-1]) / values[i-1], 4))
        else:
            growth.append(None)

    return growth

# ============================================================================
# DEAL METRICS EXTRACTION
# ============================================================================

def extract_deal_metrics(ocr_text: str, tables: List[pd.DataFrame], financials: Dict) -> Dict[str, Any]:
    """Extract deal-related metrics."""
    deal = {}

    # LTM metrics (last column of historical financials)
    hist_rev = financials.get('net_revenue_hist', [])
    hist_ebitda = financials.get('adj_ebitda_hist', [])

    deal['revenue_ltm'] = hist_rev[-1] if hist_rev and hist_rev[-1] is not None else None
    deal['ebitda_ltm'] = hist_ebitda[-1] if hist_ebitda and hist_ebitda[-1] is not None else None
    deal['ebitda_for_price'] = deal['ebitda_ltm']  # Typically LTM EBITDA

    # Entry multiple (search in text)
    deal['entry_multiple'] = extract_numeric_from_text(ocr_text, ENTRY_MULTIPLE_PATTERNS)

    # Purchase price
    purchase_price = extract_numeric_from_text(ocr_text, PURCHASE_PRICE_PATTERNS)
    deal['purchase_price_calculated'] = purchase_price
    deal['enterprise_value'] = purchase_price  # Typically same in CIMs

    # Calculate entry multiple if not found
    if deal['entry_multiple'] is None and purchase_price and deal['ebitda_for_price']:
        deal['entry_multiple'] = round(purchase_price / deal['ebitda_for_price'], 2)

    # Other deal terms
    deal['pct_acquired'] = 1.0  # Usually 100% in PE deals
    deal['exit_multiple'] = None  # Rarely disclosed
    deal['term_loan_amount'] = 0
    deal['seller_note_amount'] = 0
    deal['earnout_amount'] = 0
    deal['equity_rollover'] = 0
    deal['leverage_ratio'] = 0.0

    return deal

def extract_numeric_from_text(text: str, patterns: List[str]) -> Optional[float]:
    """Extract numeric value following a pattern in free text."""
    for pattern in patterns:
        # Look for pattern followed by number
        match = re.search(pattern + r'[:\s]+\$?\s*([\d,\.]+)(?:M|x)?', text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                value = float(num_str)
                # If it's a multiple (typically < 20), return as-is
                # If it's a dollar amount in millions, convert to thousands
                if value > 100:
                    return value  # Already in thousands or explicit value
                else:
                    return value  # Likely a multiple
            except:
                continue

    return None

# ============================================================================
# COLLATERAL EXTRACTION
# ============================================================================

def extract_collateral(balance_sheet: Optional[pd.DataFrame], tables: List[pd.DataFrame]) -> Dict[str, Any]:
    """Extract collateral values from balance sheet."""
    collateral = {}

    # Extract A/R
    ar_value = None
    if balance_sheet is not None:
        ar_row = extract_by_pattern(balance_sheet, AR_PATTERNS, 1)
        ar_value = ar_row[0] if ar_row else None

    # Extract Inventory
    inventory_value = None
    if balance_sheet is not None:
        inv_row = extract_by_pattern(balance_sheet, INVENTORY_PATTERNS, 1)
        inventory_value = inv_row[0] if inv_row else None

    # Extract Equipment/PP&E
    equipment_value = None
    if balance_sheet is not None:
        equip_row = extract_by_pattern(balance_sheet, EQUIPMENT_PATTERNS, 1)
        equipment_value = equip_row[0] if equip_row else None

    collateral['ar_value'] = ar_value
    collateral['ar_advance_rate'] = 0.75  # Standard assumption
    collateral['inventory_value'] = inventory_value
    collateral['inventory_advance_rate'] = 0.70  # Standard assumption
    collateral['equipment_value'] = equipment_value
    collateral['equipment_advance_rate'] = 0.0  # Typically not included
    collateral['building_land_value'] = None
    collateral['building_advance_rate'] = 0.0

    # Calculate ABL availability
    abl_availability = 0.0
    if ar_value:
        abl_availability += ar_value * collateral['ar_advance_rate']
    if inventory_value:
        abl_availability += inventory_value * collateral['inventory_advance_rate']

    collateral['abl_availability_calculated'] = round(abl_availability, 2) if abl_availability > 0 else None

    return collateral

# ============================================================================
# RATES & FEES EXTRACTION
# ============================================================================

def extract_rates(ocr_text: str) -> Dict[str, Any]:
    """Extract interest rates."""
    rates = {}

    rates['abl_rate'] = extract_percentage_from_text(ocr_text, ABL_RATE_PATTERNS) or 0.0675
    rates['term_rate'] = extract_percentage_from_text(ocr_text, TERM_RATE_PATTERNS) or 0.07
    rates['seller_note_rate'] = 0.05  # Standard assumption
    rates['tax_rate'] = 0.30  # Standard assumption
    rates['term_amort_years'] = 3  # Standard
    rates['seller_note_amort_years'] = 4  # Standard

    return rates

def extract_percentage_from_text(text: str, patterns: List[str]) -> Optional[float]:
    """Extract percentage value from text."""
    for pattern in patterns:
        match = re.search(pattern + r'[:\s]+([\d\.]+)%', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) / 100.0
            except:
                continue
    return None

def extract_fees(ocr_text: str) -> Dict[str, Any]:
    """Extract transaction fees (usually standard assumptions)."""
    return {
        'abl_fee_rate': 0.0075,
        'term_fee_rate': 0.0075,
        'legal_fees': 250,
        'qofe_fees': 125,
        'tax_fees': 50,
        'rw_insurance': 75,
        'bonus_senior': 300,
        'bonus_junior': 100,
    }

# ============================================================================
# QUALITATIVE DATA EXTRACTION
# ============================================================================

def extract_qualitative(ocr_text: str) -> Dict[str, Any]:
    """Extract qualitative highlights and risks."""
    qualitative = {}

    # Extract key highlights (look for bullet points after "Highlights" or "Investment Highlights")
    highlights = []
    highlights_section = re.search(r'(?:Investment\s+)?Highlights:?\s*\n((?:[•\-\*]\s*.+\n?)+)', ocr_text, re.IGNORECASE | re.MULTILINE)
    if highlights_section:
        bullet_points = re.findall(r'[•\-\*]\s*(.+)', highlights_section.group(1))
        highlights = [bp.strip() for bp in bullet_points[:5]]  # Top 5

    qualitative['key_highlights'] = highlights
    qualitative['risks'] = []  # Rarely disclosed in CIMs
    qualitative['company_summary'] = None

    return qualitative

# ============================================================================
# VALIDATION
# ============================================================================

def validate_extraction(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate extracted data against business rules and accounting principles.
    Returns (corrected_data, warnings).
    """
    warnings = []
    financials = data.get('financials', {})

    # 1. Revenue > Gross Profit (ALWAYS)
    rev_hist = financials.get('net_revenue_hist', [])
    gp_hist = financials.get('gross_profit_hist', [])

    for i, (rev, gp) in enumerate(zip(rev_hist, gp_hist)):
        if rev is not None and gp is not None:
            if rev < gp:
                warnings.append(f"CRITICAL: Year {i} Revenue ({rev}) < Gross Profit ({gp}) - IMPOSSIBLE")
                # Attempt swap if GP looks like it could be revenue
                if gp > rev * 1.5:
                    rev_hist[i], gp_hist[i] = gp, rev
                    warnings.append(f"AUTO-CORRECTED: Swapped Revenue and Gross Profit for year {i}")

    # 2. EBITDA > Operating Income (if Operating Income extracted)
    ebitda_hist = financials.get('adj_ebitda_hist', [])
    # (Operating Income not extracted in current implementation)

    # 3. Reasonable margins
    for i, (rev, gp) in enumerate(zip(rev_hist, gp_hist)):
        if rev and gp and rev > 0:
            gm = gp / rev
            if gm < 0.05 or gm > 0.95:
                warnings.append(f"Year {i}: Gross margin {gm:.1%} outside typical range (5-95%)")

    for i, (rev, ebitda) in enumerate(zip(rev_hist, ebitda_hist)):
        if rev and ebitda and rev > 0:
            em = ebitda / rev
            if em < 0.00 or em > 0.60:
                warnings.append(f"Year {i}: EBITDA margin {em:.1%} outside typical range (0-60%)")

    # 4. Array length consistency
    hist_years = len(data.get('historical_years', []))
    for field in ['net_revenue_hist', 'gross_profit_hist', 'adj_ebitda_hist']:
        values = financials.get(field, [])
        if len(values) != hist_years:
            warnings.append(f"{field}: Expected {hist_years} values, got {len(values)}")

    # 5. CapEx and Mgmt Fees should be negative
    capex_hist = financials.get('capex_hist', [])
    for i, val in enumerate(capex_hist):
        if val is not None and val > 0:
            warnings.append(f"CapEx hist[{i}] is positive ({val}) - should be negative")

    mgmt_fees = financials.get('mgmt_fees_proj', [])
    for i, val in enumerate(mgmt_fees):
        if val is not None and val > 0:
            warnings.append(f"Mgmt Fees proj[{i}] is positive ({val}) - should be negative or zero")

    return data, warnings

def apply_business_rules(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply business rules to derive missing values or correct inconsistencies.
    """
    financials = data.get('financials', {})

    # Rule 1: If SG&A is missing, try to derive from GP - EBITDA - D&A
    sga_hist = financials.get('sga_hist', [])
    gp_hist = financials.get('gross_profit_hist', [])
    ebitda_hist = financials.get('adj_ebitda_hist', [])
    da_hist = financials.get('depreciation_hist', [])
    adj_hist = financials.get('adjustments_hist', [])

    for i in range(len(sga_hist)):
        if sga_hist[i] is None:
            # SG&A = GP - EBITDA - D&A + Adjustments (approximately)
            if all(v is not None for v in [gp_hist[i], ebitda_hist[i]]):
                da = da_hist[i] if da_hist[i] is not None else 0
                adj = adj_hist[i] if adj_hist[i] is not None else 0
                sga_hist[i] = gp_hist[i] - ebitda_hist[i] - da + adj

    # Rule 2: Pad projection arrays to exactly 5 years
    proj_years = data.get('projection_years', [])
    if len(proj_years) < 5:
        # Pad with last year + incrementing
        last_year = proj_years[-1] if proj_years else "Dec-26F"
        while len(proj_years) < 5:
            # Increment year
            match = re.search(r'(\d{2,4})([FEP])', last_year)
            if match:
                year = int(match.group(1))
                suffix = match.group(2)
                year += 1
                if year > 99 and year < 2000:  # Handle 2-digit years
                    year += 2000
                proj_years.append(f"Dec-{year}{suffix}")
            else:
                proj_years.append(None)

        data['projection_years'] = proj_years

    # Pad all projection arrays
    for key in ['net_revenue_proj', 'gross_profit_proj', 'sga_proj', 'adj_ebitda_proj',
                'adjustments_proj', 'depreciation_proj', 'capex_proj', 'mgmt_fees_proj']:
        arr = financials.get(key, [])
        while len(arr) < 5:
            arr.append(None)
        financials[key] = arr[:5]  # Truncate if > 5

    return data

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_confidence(extraction_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate confidence scores for extracted data."""
    if extraction_data is None:
        return {
            "overall_confidence": 0,
            "deal_overview_confidence": 0,
            "financial_summary_confidence": 0,
            "deal_metrics_confidence": 0,
            "collateral_confidence": 0,
            "projections_confidence": 0,
            "field_level": {}
        }

    financials = extraction_data.get('financials', {})

    # Field-level confidence based on whether values were extracted
    field_level = {}

    def get_confidence(values):
        if values is None:
            return "not_found"
        if all(v is not None for v in values):
            return "high"
        if any(v is not None for v in values):
            return "medium"
        return "not_found"

    field_level['net_revenue'] = get_confidence(financials.get('net_revenue_hist'))
    field_level['gross_profit'] = get_confidence(financials.get('gross_profit_hist'))
    field_level['sga'] = get_confidence(financials.get('sga_hist'))
    field_level['adj_ebitda'] = get_confidence(financials.get('adj_ebitda_hist'))
    field_level['depreciation'] = get_confidence(financials.get('depreciation_hist'))
    field_level['capex'] = get_confidence(financials.get('capex_hist'))
    field_level['adjustments'] = get_confidence(financials.get('adjustments_hist'))
    field_level['projections'] = get_confidence(financials.get('net_revenue_proj'))

    # Overall scores (percentage)
    deal_overview_conf = 70 if extraction_data.get('company_name') else 0
    financial_conf = sum(1 for v in [field_level.get(k) for k in ['net_revenue', 'adj_ebitda', 'gross_profit']] if v == "high") / 3 * 100
    deal_conf = 70 if extraction_data.get('deal', {}).get('ebitda_ltm') else 0
    collateral_conf = 70 if extraction_data.get('collateral', {}).get('ar_value') else 0
    projections_conf = 85 if field_level.get('projections') == "high" else 50

    overall_conf = (deal_overview_conf + financial_conf + deal_conf + collateral_conf + projections_conf) / 5

    return {
        "overall_confidence": round(overall_conf),
        "deal_overview_confidence": deal_overview_conf,
        "financial_summary_confidence": round(financial_conf),
        "deal_metrics_confidence": deal_conf,
        "collateral_confidence": collateral_conf,
        "projections_confidence": projections_conf,
        "field_level": field_level
    }
