import os
import json
import uuid
import threading
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from database import get_db, init_db
from services.ocr_service import process_document_ocr
from services.rule_based_extraction_service import extract_document_financials

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), os.getenv('UPLOAD_FOLDER', 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE_MB', 50)) * 1024 * 1024

# Always initialize the database on startup
init_db()


# ---------- Serve Frontend ----------

@app.route('/')
def index():
    return render_template('index.html')


# ---------- Dashboard API ----------

@app.route('/api/dashboard')
def dashboard():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) as total FROM documents')
    total = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE ocr_status = 'completed'")
    analyzed = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE ocr_status IN ('pending', 'processing')")
    pending = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE ocr_status = 'failed'")
    failed = cursor.fetchone()['count']

    cursor.execute('SELECT AVG(ocr_confidence) as avg_conf FROM documents WHERE ocr_confidence > 0')
    row = cursor.fetchone()
    accuracy = round((row['avg_conf'] or 0) * 100, 1)

    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE extraction_status = 'completed'")
    fully_analyzed = cursor.fetchone()['count']

    cursor.execute('SELECT AVG(confidence_score) as avg_conf FROM documents WHERE confidence_score > 0')
    row = cursor.fetchone()
    avg_extraction_confidence = round((row['avg_conf'] or 0) * 100, 1)

    cursor.execute('SELECT AVG(entry_multiple) as avg_mult FROM documents WHERE entry_multiple IS NOT NULL')
    row = cursor.fetchone()
    avg_entry_multiple = round(row['avg_mult'] or 0, 1)

    cursor.execute('SELECT SUM(purchase_price) as total_val FROM documents WHERE purchase_price IS NOT NULL')
    row = cursor.fetchone()
    total_deal_value = row['total_val'] or 0

    cursor.execute(
        'SELECT original_name, upload_date, status, ocr_status, extraction_status, company_name FROM documents ORDER BY upload_date DESC LIMIT 10'
    )
    recent = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return jsonify({
        'total': total,
        'analyzed': analyzed,
        'pending': pending,
        'failed': failed,
        'accuracy': accuracy,
        'fully_analyzed': fully_analyzed,
        'avg_extraction_confidence': avg_extraction_confidence,
        'avg_entry_multiple': avg_entry_multiple,
        'total_deal_value': total_deal_value,
        'recent': recent
    })


# ---------- Upload API ----------

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    uploaded = []

    conn = get_db()
    cursor = conn.cursor()

    for file in files:
        if file.filename == '':
            continue

        # Generate unique filename to avoid collisions
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        size = os.path.getsize(filepath)

        cursor.execute(
            'INSERT INTO documents (filename, original_name, size) VALUES (?, ?, ?)',
            (unique_name, file.filename, size)
        )

        doc_id = cursor.lastrowid
        conn.commit()

        # Launch OCR in background thread
        thread = threading.Thread(
            target=process_document_ocr,
            args=(doc_id, filepath),
            daemon=True
        )
        thread.start()

        uploaded.append({
            'id': doc_id,
            'original_name': file.filename,
            'size': size,
            'status': 'processing',
            'message': 'File uploaded. OCR processing started.'
        })

    conn.close()

    return jsonify({'uploaded': uploaded}), 201


# ---------- Documents API ----------

@app.route('/api/documents')
def get_documents():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents ORDER BY upload_date DESC')
    docs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(docs)


@app.route('/api/documents/<int:doc_id>/status')
def get_document_status(doc_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    return jsonify({
        'document_id': row['id'],
        'filename': row['original_name'],
        'status': row['status'],
        'ocr_status': row['ocr_status'],
        'ocr_confidence': row['ocr_confidence'],
        'page_count': row['page_count'],
        'word_count': row['word_count'],
        'ocr_completed_at': row['ocr_completed_at'],
        'error_message': row['error_message'],
        'extraction_status': row['extraction_status'] or 'pending',
        'extraction_error': row['extraction_error'],
        'company_name': row['company_name'],
        'confidence_score': row['confidence_score'],
    })


@app.route('/api/documents/<int:doc_id>/text')
def get_document_text(doc_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    if row['ocr_status'] != 'completed':
        return jsonify({
            'status': row['ocr_status'],
            'message': 'OCR not yet complete'
        }), 202

    text_path = os.path.join(os.path.dirname(__file__), row['processed_text_path'])
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        return jsonify({'error': 'Processed text file not found'}), 404

    return jsonify({
        'document_id': row['id'],
        'filename': row['original_name'],
        'text': text_content,
        'word_count': row['word_count'],
        'page_count': row['page_count'],
    })


@app.route('/api/documents/<int:doc_id>/extraction')
def get_document_extraction(doc_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    if row['extraction_status'] != 'completed':
        return jsonify({
            'extraction_status': row['extraction_status'] or 'pending',
            'extraction_error': row['extraction_error'],
            'message': 'Extraction not yet complete'
        }), 202

    # Read the full extraction JSON
    extraction_path = os.path.join(os.path.dirname(__file__), row['extraction_path'])
    try:
        with open(extraction_path, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'Extraction file not found'}), 404

    return jsonify({
        'document_id': row['id'],
        'filename': row['original_name'],
        'extraction_status': row['extraction_status'],
        'extraction_completed_at': row['extraction_completed_at'],
        'confidence_score': row['confidence_score'],
        'data': extraction_data,
    })


@app.route('/api/documents/<int:doc_id>/analysis')
def get_document_analysis(doc_id):
    """Combined endpoint returning OCR + extraction data for analysis view."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    result = {
        'document_id': row['id'],
        'filename': row['original_name'],
        'size': row['size'],
        'upload_date': row['upload_date'],
        'ocr_status': row['ocr_status'],
        'ocr_confidence': row['ocr_confidence'],
        'page_count': row['page_count'],
        'word_count': row['word_count'],
        'extraction_status': row['extraction_status'] or 'pending',
        'extraction_error': row['extraction_error'],
        'company_name': row['company_name'],
        'confidence_score': row['confidence_score'],
        'extraction': None,
    }

    if row['extraction_status'] == 'completed' and row['extraction_path']:
        extraction_path = os.path.join(os.path.dirname(__file__), row['extraction_path'])
        try:
            with open(extraction_path, 'r', encoding='utf-8') as f:
                extraction = json.load(f)
            result['extraction'] = extraction
            # Flatten key fields to top level for convenience
            result['company_name'] = extraction.get('company_name') or row['company_name']
            result['industry'] = extraction.get('industry')
            result['geography'] = extraction.get('geography')
            result['historical_years'] = extraction.get('historical_years', [])
            result['projection_years'] = extraction.get('projection_years', [])
        except FileNotFoundError:
            pass

    return jsonify(result)


@app.route('/api/documents/<int:doc_id>/re-extract', methods=['POST'])
def re_extract_document(doc_id):
    """Re-run AI extraction on a document that already has OCR completed."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    if row['ocr_status'] != 'completed' or not row['processed_text_path']:
        return jsonify({'error': 'OCR not completed for this document'}), 400

    # Launch extraction in background thread
    thread = threading.Thread(
        target=extract_document_financials,
        args=(doc_id, row['processed_text_path']),
        daemon=True
    )
    thread.start()

    return jsonify({'message': f'Re-extraction started for document {doc_id}'}), 202


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT filename, raw_ocr_path, processed_text_path, extraction_path FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return jsonify({'error': 'Document not found'}), 404

    # Delete uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, row['filename'])
    if os.path.exists(filepath):
        os.remove(filepath)

    # Delete OCR files if they exist
    base_dir = os.path.dirname(__file__)
    if row['raw_ocr_path']:
        raw_path = os.path.join(base_dir, row['raw_ocr_path'])
        if os.path.exists(raw_path):
            os.remove(raw_path)
    if row['processed_text_path']:
        proc_path = os.path.join(base_dir, row['processed_text_path'])
        if os.path.exists(proc_path):
            os.remove(proc_path)
    if row['extraction_path']:
        ext_path = os.path.join(base_dir, row['extraction_path'])
        if os.path.exists(ext_path):
            os.remove(ext_path)

    cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Document deleted'})


# ---------- Accuracy API ----------

CHIMERA_GROUND_TRUTH = {
    'net_revenue_hist': [125837, 99086, 92452],
    'gross_profit_hist': [54074, 40680, 36843],
    'sga_hist': [32847, 31559, 28812],
    'adjustments_hist': [406, 1797, 549],
    'adj_ebitda_hist': [21633, 10918, 8580],
    'depreciation_hist': [2604, 2438, 2305],
    'capex_hist': [-535, -589, -424],
    'net_revenue_proj': [96100, 108566, 123168, 123168, 123168],
    'adj_ebitda_proj': [10825, 15124, 20895, 19395, 19395],
    'depreciation_proj': [2063, 1968, 1872, 1872, 1872],
    'capex_proj': [-1400, -1000, -600, -600, -600],
    'mgmt_fees_proj': [-2000, -2500, -3000, -5000, -5000],
    'entry_multiple': 3.0,
    'purchase_price': 25743,
    'ebitda_for_price': 8581,
    'revenue_ltm': 92452,
    'ebitda_ltm': 8580,
    'ar_value': 6147,
    'inventory_value': 13512,
    'abl_rate': 0.0675,
    'term_rate': 0.07,
}


def _compare_field(extracted, expected, tolerance=0.05):
    """Compare a single field value against expected with tolerance."""
    if expected is None:
        return {'status': 'SKIP', 'reason': 'no ground truth'}

    if extracted is None:
        return {'status': 'FAIL', 'extracted': None, 'expected': expected, 'error': 'not extracted'}

    if isinstance(expected, list):
        if not isinstance(extracted, list):
            return {'status': 'FAIL', 'extracted': extracted, 'expected': expected, 'error': 'not a list'}
        results = []
        all_pass = True
        for i, (ext_v, exp_v) in enumerate(zip(extracted, expected)):
            if exp_v is None:
                results.append({'index': i, 'status': 'SKIP'})
                continue
            if ext_v is None:
                results.append({'index': i, 'status': 'FAIL', 'error': 'null'})
                all_pass = False
                continue
            if exp_v == 0:
                if ext_v == 0:
                    results.append({'index': i, 'status': 'PASS'})
                else:
                    results.append({'index': i, 'status': 'FAIL', 'extracted': ext_v, 'expected': exp_v})
                    all_pass = False
                continue
            pct_err = abs(ext_v - exp_v) / abs(exp_v)
            if pct_err <= tolerance:
                results.append({'index': i, 'status': 'PASS', 'error_pct': f'{pct_err:.1%}'})
            else:
                results.append({'index': i, 'status': 'FAIL', 'extracted': ext_v, 'expected': exp_v, 'error_pct': f'{pct_err:.1%}'})
                all_pass = False
        return {'status': 'PASS' if all_pass else 'FAIL', 'extracted': extracted, 'expected': expected, 'details': results}
    else:
        if expected == 0:
            status = 'PASS' if extracted == 0 else 'FAIL'
            return {'status': status, 'extracted': extracted, 'expected': expected}
        pct_err = abs(extracted - expected) / abs(expected)
        status = 'PASS' if pct_err <= tolerance else 'FAIL'
        return {'status': status, 'extracted': extracted, 'expected': expected, 'error_pct': f'{pct_err:.1%}'}


@app.route('/api/documents/<int:doc_id>/accuracy')
def get_document_accuracy(doc_id):
    """Compare extraction to known ground truth (Chimera hardcoded)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'Document not found'}), 404

    if row['extraction_status'] != 'completed' or not row['extraction_path']:
        return jsonify({'error': 'Extraction not completed'}), 400

    extraction_path = os.path.join(os.path.dirname(__file__), row['extraction_path'])
    try:
        with open(extraction_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'Extraction file not found'}), 404

    fin = ext.get('financials', {})
    deal = ext.get('deal', {})
    coll = ext.get('collateral', {})
    rates_data = ext.get('rates', {})

    field_map = {
        'net_revenue_hist': fin.get('net_revenue_hist'),
        'gross_profit_hist': fin.get('gross_profit_hist'),
        'sga_hist': fin.get('sga_hist'),
        'adjustments_hist': fin.get('adjustments_hist'),
        'adj_ebitda_hist': fin.get('adj_ebitda_hist'),
        'depreciation_hist': fin.get('depreciation_hist'),
        'capex_hist': fin.get('capex_hist'),
        'net_revenue_proj': fin.get('net_revenue_proj'),
        'adj_ebitda_proj': fin.get('adj_ebitda_proj'),
        'depreciation_proj': fin.get('depreciation_proj'),
        'capex_proj': fin.get('capex_proj'),
        'mgmt_fees_proj': fin.get('mgmt_fees_proj'),
        'entry_multiple': deal.get('entry_multiple'),
        'purchase_price': deal.get('purchase_price_calculated'),
        'ebitda_for_price': deal.get('ebitda_for_price'),
        'revenue_ltm': deal.get('revenue_ltm'),
        'ebitda_ltm': deal.get('ebitda_ltm'),
        'ar_value': coll.get('ar_value'),
        'inventory_value': coll.get('inventory_value'),
        'abl_rate': rates_data.get('abl_rate'),
        'term_rate': rates_data.get('term_rate'),
    }

    field_accuracy = {}
    fields_checked = 0
    fields_correct = 0

    for field_name, expected in CHIMERA_GROUND_TRUTH.items():
        extracted = field_map.get(field_name)
        result = _compare_field(extracted, expected)
        field_accuracy[field_name] = result
        if result['status'] != 'SKIP':
            fields_checked += 1
            if result['status'] == 'PASS':
                fields_correct += 1

    accuracy_score = round((fields_correct / fields_checked * 100), 1) if fields_checked > 0 else 0

    return jsonify({
        'accuracy_score': accuracy_score,
        'fields_checked': fields_checked,
        'fields_correct': fields_correct,
        'fields_wrong': fields_checked - fields_correct,
        'field_accuracy': field_accuracy,
        'corrections_applied': ext.get('_corrections_applied', []),
    })


# ---------- Settings API ----------

@app.route('/api/settings')
def get_settings():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT key, value FROM settings')
    settings = {row['key']: row['value'] for row in cursor.fetchall()}
    conn.close()
    return jsonify(settings)


@app.route('/api/settings', methods=['PUT'])
def update_settings():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    conn = get_db()
    cursor = conn.cursor()

    for key, value in data.items():
        cursor.execute(
            'INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)',
            (key, str(value))
        )

    conn.commit()
    conn.close()

    return jsonify({'message': 'Settings updated'})


# ---------- Start Server ----------

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
