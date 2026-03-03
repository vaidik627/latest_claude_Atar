import os
import io
import time
from datetime import datetime, timezone

from google.cloud import documentai_v1 as documentai
from google.protobuf.json_format import MessageToJson
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv

import threading
from database import update_document_ocr_status

load_dotenv()

PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
PROCESSOR_ID = os.getenv('GOOGLE_PROCESSOR_ID')
LOCATION = os.getenv('GOOGLE_LOCATION', 'us')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Resolve credentials path to absolute if relative
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
if creds_path and not os.path.isabs(creds_path):
    abs_creds = os.path.join(BASE_DIR, creds_path)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = abs_creds

OCR_RAW_FOLDER = os.path.join(BASE_DIR, os.getenv('OCR_RAW_FOLDER', 'storage/ocr/raw'))
OCR_PROCESSED_FOLDER = os.path.join(BASE_DIR, os.getenv('OCR_PROCESSED_FOLDER', 'storage/ocr/processed'))

MAX_INLINE_SIZE = 15 * 1024 * 1024  # 15MB
MAX_PAGES_PER_REQUEST = 15

os.makedirs(OCR_RAW_FOLDER, exist_ok=True)
os.makedirs(OCR_PROCESSED_FOLDER, exist_ok=True)


def process_document_ocr(doc_id, pdf_path):
    """Main OCR processing function. Runs in a background thread."""
    try:
        # a) Mark as processing
        update_document_ocr_status(doc_id, 'processing')
        print(f"[OCR] doc_id={doc_id}: Processing started")

        # b) Read PDF and count pages
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"[OCR] doc_id={doc_id}: PDF has {total_pages} pages")

        # c) Split into chunks if needed, process each
        all_documents = []
        all_raw_json = []

        for chunk_start in range(0, total_pages, MAX_PAGES_PER_REQUEST):
            chunk_end = min(chunk_start + MAX_PAGES_PER_REQUEST, total_pages)
            print(f"[OCR] doc_id={doc_id}: Processing pages {chunk_start + 1}-{chunk_end}")

            # Extract chunk as PDF bytes
            writer = PdfWriter()
            for i in range(chunk_start, chunk_end):
                writer.add_page(reader.pages[i])

            chunk_buffer = io.BytesIO()
            writer.write(chunk_buffer)
            chunk_bytes = chunk_buffer.getvalue()

            # Check size
            if len(chunk_bytes) > MAX_INLINE_SIZE:
                raise ValueError(
                    f"PDF chunk (pages {chunk_start + 1}-{chunk_end}) is {len(chunk_bytes)} bytes, "
                    "exceeds 15MB inline limit. GCS-based processing not yet supported."
                )

            # Call Document AI
            document = _call_document_ai_with_retry(chunk_bytes)
            all_documents.append((chunk_start, document))
            all_raw_json.append(MessageToJson(document._pb))

        # e) Save raw JSON response (all chunks)
        raw_path = os.path.join(OCR_RAW_FOLDER, f'{doc_id}.json')
        with open(raw_path, 'w', encoding='utf-8') as f:
            if len(all_raw_json) == 1:
                f.write(all_raw_json[0])
            else:
                f.write('[\n' + ',\n'.join(all_raw_json) + '\n]')

        # f) Extract clean text from all chunks
        all_text_parts = []
        total_confidence = 0.0
        confidence_count = 0

        for chunk_start, document in all_documents:
            for page in document.pages:
                # Adjust page number for chunked processing
                actual_page_num = chunk_start + page.page_number
                page_parts = [f"=== PAGE {actual_page_num} ==="]

                table_texts = []
                for table in page.tables:
                    table_text = _format_table(table, document.text)
                    table_texts.append(f"[TABLE - Page {actual_page_num}]\n{table_text}")

                for paragraph in page.paragraphs:
                    text = _get_layout_text(paragraph.layout, document.text)
                    if text.strip():
                        page_parts.append(text.strip())

                for tt in table_texts:
                    page_parts.append(tt)

                all_text_parts.append('\n\n'.join(page_parts))

            # Accumulate confidence from this chunk
            for page in document.pages:
                for token in page.tokens:
                    conf = token.layout.confidence
                    if conf > 0:
                        total_confidence += conf
                        confidence_count += 1

        extracted_text = '\n\n'.join(all_text_parts)

        # g) Save processed text
        processed_path = os.path.join(OCR_PROCESSED_FOLDER, f'{doc_id}.txt')
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        # h) Calculate stats
        page_count = total_pages
        word_count = len(extracted_text.split())
        ocr_confidence = (total_confidence / confidence_count) if confidence_count > 0 else 0.0

        # i) Update DB — completed
        update_document_ocr_status(
            doc_id, 'completed',
            status='analyzed',
            ocr_completed_at=datetime.now(timezone.utc).isoformat(),
            raw_ocr_path=os.path.join('storage', 'ocr', 'raw', f'{doc_id}.json'),
            processed_text_path=os.path.join('storage', 'ocr', 'processed', f'{doc_id}.txt'),
            page_count=page_count,
            word_count=word_count,
            ocr_confidence=round(ocr_confidence, 4),
        )
        print(f"[OCR] doc_id={doc_id}: Completed — {page_count} pages, {word_count} words, confidence={ocr_confidence:.2%}")

        # Chain AI extraction after successful OCR
        processed_text_rel = os.path.join('storage', 'ocr', 'processed', f'{doc_id}.txt')
        _start_extraction(doc_id, processed_text_rel)

    except Exception as e:
        # j) Error handling
        print(f"[OCR ERROR] doc_id={doc_id}: {e}")
        update_document_ocr_status(
            doc_id, 'failed',
            status='failed',
            error_message=str(e),
        )


def _call_document_ai_with_retry(pdf_bytes, max_retries=3):
    """Call Google Document AI with exponential backoff retry."""
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(
            content=pdf_bytes,
            mime_type='application/pdf',
        ),
    )

    last_exception = None
    for attempt in range(max_retries):
        try:
            result = client.process_document(request=request)
            return result.document
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = 2 ** (attempt + 1)  # 2s, 4s, 8s
                print(f"[OCR RETRY] attempt {attempt + 1}/{max_retries}, retrying in {delay}s: {e}")
                time.sleep(delay)

    raise last_exception


def _format_table(table, full_text):
    """Format a Document AI table as a markdown pipe-delimited table."""
    rows = []

    if table.header_rows:
        for header_row in table.header_rows:
            cells = []
            for cell in header_row.cells:
                cell_text = _get_layout_text(cell.layout, full_text).strip().replace('\n', ' ')
                cells.append(cell_text)
            rows.append('| ' + ' | '.join(cells) + ' |')
            rows.append('| ' + ' | '.join('------' for _ in cells) + ' |')

    for body_row in table.body_rows:
        cells = []
        for cell in body_row.cells:
            cell_text = _get_layout_text(cell.layout, full_text).strip().replace('\n', ' ')
            cells.append(cell_text)
        rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(rows)


def _get_layout_text(layout, full_text):
    """Extract text from a layout element using text anchors."""
    if not layout.text_anchor or not layout.text_anchor.text_segments:
        return ''

    parts = []
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index)
        parts.append(full_text[start:end])

    return ''.join(parts)


def _start_extraction(doc_id, processed_text_path):
    """Launch rule-based extraction in a new background thread."""
    from services.rule_based_extraction_service import extract_document_financials
    thread = threading.Thread(
        target=extract_document_financials,
        args=(doc_id, processed_text_path),
        daemon=True
    )
    thread.start()
    print(f"[OCR] doc_id={doc_id}: Rule-based extraction thread started")
