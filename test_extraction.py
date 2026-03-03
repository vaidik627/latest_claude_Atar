"""
Quick test script for rule-based extraction
"""
import sys
import json
from services.rule_based_extraction_service import extract_document_financials

def test_extraction(doc_id):
    """Test extraction on a specific document"""
    ocr_path = f"storage/ocr/processed/{doc_id}.txt"

    print(f"\n{'='*60}")
    print(f"Testing Rule-Based Extraction on Document {doc_id}")
    print(f"{'='*60}\n")

    try:
        # Run extraction
        extract_document_financials(doc_id, ocr_path)

        # Load and display results
        result_path = f"storage/extractions/{doc_id}.json"
        with open(result_path, 'r') as f:
            result = json.load(f)

        print("\n✅ Extraction completed successfully!")
        print(f"\nCompany: {result.get('company_name')}")
        print(f"Industry: {result.get('industry')}")
        print(f"Historical Years: {result.get('historical_years')}")
        print(f"Projection Years: {result.get('projection_years')}")

        print("\n📊 Financial Data:")
        financials = result.get('financials', {})
        print(f"  Revenue (Historical): {financials.get('net_revenue_hist')}")
        print(f"  Gross Profit (Historical): {financials.get('gross_profit_hist')}")
        print(f"  EBITDA (Historical): {financials.get('adj_ebitda_hist')}")
        print(f"  Revenue (Projections): {financials.get('net_revenue_proj')}")
        print(f"  EBITDA (Projections): {financials.get('adj_ebitda_proj')}")

        print("\n💰 Deal Metrics:")
        deal = result.get('deal', {})
        print(f"  Revenue LTM: {deal.get('revenue_ltm')}")
        print(f"  EBITDA LTM: {deal.get('ebitda_ltm')}")
        print(f"  Entry Multiple: {deal.get('entry_multiple')}")
        print(f"  Purchase Price: {deal.get('purchase_price_calculated')}")

        print("\n⚠️  Validation Warnings:")
        warnings = result.get('_validation_warnings', [])
        if warnings:
            for w in warnings[:10]:  # Show first 10
                print(f"  - {w}")
        else:
            print("  ✅ No warnings!")

        print(f"\n📈 Overall Confidence: {result.get('confidence', {}).get('overall_confidence')}%")

        print("\n" + "="*60)
        print("Test completed!")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    doc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    success = test_extraction(doc_id)
    sys.exit(0 if success else 1)
