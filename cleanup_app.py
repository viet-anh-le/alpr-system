import os
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

BASE_DIR = '/home/vietanh/Documents/DATN/ALPR_Vietnamese'
OCR_IMAGES_DIR = os.path.join(BASE_DIR, 'data/raw/platesmania_vn/ocr/images')
OCR_LABELS_DIR = os.path.join(BASE_DIR, 'data/raw/platesmania_vn/ocr/labels')
DET_IMAGES_DIR = os.path.join(BASE_DIR, 'data/raw/platesmania_vn/detection/images')
DET_LABELS_DIR = os.path.join(BASE_DIR, 'data/raw/platesmania_vn/detection/labels')

CSV_TRAIN = os.path.join(BASE_DIR, 'runs/infer/quality_router_platesmania_vn_train/predictions.csv')
CSV_VAL = os.path.join(BASE_DIR, 'runs/infer/quality_router_platesmania_vn_val/predictions_val.csv')

# Load CSVs as global variables
df_train = pd.read_csv(CSV_TRAIN) if os.path.exists(CSV_TRAIN) else pd.DataFrame()
df_val = pd.read_csv(CSV_VAL) if os.path.exists(CSV_VAL) else pd.DataFrame()

# Build lookup dict
pred_lookup = {}
def build_lookup():
    global pred_lookup
    pred_lookup.clear()
    for df in [df_train, df_val]:
        if not df.empty:
            for _, row in df.iterrows():
                if 'path' in row:
                    basename = os.path.basename(str(row['path']))
                    pred_lookup[basename] = {
                        'legibility': row.get('predicted_legibility', 'Unknown'),
                        'conf': row.get('router_conf', 0.0),
                        'route': row.get('route', 'Unknown'),
                        'quality_score': row.get('quality_score', 0.0)
                    }

build_lookup()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dataset Cleaner & Labeler</title>
    <style>
        body { font-family: sans-serif; background: #f0f0f0; padding: 20px; }
        .controls { margin-bottom: 20px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: sticky; top: 0; z-index: 100;}
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
        .card { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        .card img { max-width: 100%; height: auto; border-radius: 4px; object-fit: contain; max-height: 150px; display: block; margin: 0 auto; cursor: pointer; }
        .info { margin-top: 10px; font-size: 14px; flex-grow: 1; }
        .info span { display: block; margin-bottom: 4px; }
        .btn-delete { margin-top: 5px; background: #dc3545; color: white; border: none; padding: 6px; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%;}
        .btn-delete:hover { background: #c82333; }
        
        .label-selector { margin-top: 5px; width: 100%; padding: 4px; border-radius: 4px; border: 1px solid #ccc; font-weight: bold;}
        .label-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; flex-wrap: wrap;}

        .label-good { color: #28a745; font-weight: bold; }
        .label-illegible { color: #dc3545; font-weight: bold; }
        .label-poor { color: #fd7e14; font-weight: bold; }
        .label-perfect { color: #20c997; font-weight: bold; }
        .label-Unknown { color: #6c757d; font-weight: bold; }
        .deleted { opacity: 0.3; pointer-events: none; }
        .pagination { margin-top: 20px; text-align: center; }
        .pagination a { display: inline-block; padding: 8px 16px; margin: 0 4px; background: white; border: 1px solid #ccc; border-radius: 4px; text-decoration: none; color: black; }
        .pagination a.active { background: #007bff; color: white; border-color: #007bff; }
        
        /* Highlight updated card briefly */
        .updated { animation: flash 1s ease-out; }
        @keyframes flash {
            0% { background-color: #d4edda; }
            100% { background-color: white; }
        }
    </style>
</head>
<body>
    <h1>Dataset Cleaner & Labeler</h1>
    <div class="controls">
        <p><strong>Total images:</strong> {{ total }} | <strong>Page {{ page }} of {{ total_pages }}</strong></p>
        <form method="get">
            <label>Filter Legibility:</label>
            <select name="filter" onchange="this.form.submit()">
                <option value="all" {% if current_filter == 'all' %}selected{% endif %}>All</option>
                <option value="good" {% if current_filter == 'good' %}selected{% endif %}>Good</option>
                <option value="perfect" {% if current_filter == 'perfect' %}selected{% endif %}>Perfect</option>
                <option value="poor" {% if current_filter == 'poor' %}selected{% endif %}>Poor</option>
                <option value="illegible" {% if current_filter == 'illegible' %}selected{% endif %}>Illegible</option>
                <option value="Unknown" {% if current_filter == 'Unknown' %}selected{% endif %}>Unknown</option>
            </select>
            <label style="margin-left: 20px;">Sort by Conf:</label>
            <select name="sort" onchange="this.form.submit()">
                <option value="none" {% if current_sort == 'none' %}selected{% endif %}>None</option>
                <option value="asc" {% if current_sort == 'asc' %}selected{% endif %}>Ascending</option>
                <option value="desc" {% if current_sort == 'desc' %}selected{% endif %}>Descending</option>
            </select>
            <input type="hidden" name="page" value="1">
        </form>
    </div>

    <div class="grid">
        {% for img in images %}
        <div class="card" id="card-{{ img.basename }}">
            <a href="/image?path={{ img.rel_path }}" target="_blank">
                <img src="/image?path={{ img.rel_path }}" loading="lazy" title="Click to view full size">
            </a>
            <div class="info">
                <div class="label-row">
                    <span id="leg-text-{{ img.basename }}" class="label-{{ img.legibility }}">Leg: {{ img.legibility }}</span>
                    <select class="label-selector" onchange="updateLabel('{{ img.basename }}', this.value)">
                        <option value="Unknown" {% if img.legibility == 'Unknown' %}selected{% endif %}>--- Change label ---</option>
                        <option value="good" {% if img.legibility == 'good' %}selected{% endif %}>good</option>
                        <option value="perfect" {% if img.legibility == 'perfect' %}selected{% endif %}>perfect</option>
                        <option value="poor" {% if img.legibility == 'poor' %}selected{% endif %}>poor</option>
                        <option value="illegible" {% if img.legibility == 'illegible' %}selected{% endif %}>illegible</option>
                    </select>
                </div>
                <span>Route: {{ img.route }}</span>
                <span>Conf: {{ "%.4f"|format(img.conf) }}</span>
                <span>Qual Score: {{ "%.4f"|format(img.quality_score) }}</span>
                <span style="font-size: 11px; color: #666; word-break: break-all;">{{ img.basename }}</span>
            </div>
            <button class="btn-delete" onclick="deleteImage('{{ img.basename }}')">Delete File</button>
        </div>
        {% endfor %}
    </div>

    <div class="pagination">
        {% if page > 1 %}
            <a href="?page=1&filter={{ current_filter }}&sort={{ current_sort }}">&laquo; First</a>
            <a href="?page={{ page - 1 }}&filter={{ current_filter }}&sort={{ current_sort }}">Previous</a>
        {% endif %}
        
        {% set start_page = max(1, page - 4) %}
        {% set end_page = min(total_pages, page + 4) %}
        
        {% for p in range(start_page, end_page + 1) %}
            <a href="?page={{ p }}&filter={{ current_filter }}&sort={{ current_sort }}" class="{% if p == page %}active{% endif %}">{{ p }}</a>
        {% endfor %}

        {% if page < total_pages %}
            <a href="?page={{ page + 1 }}&filter={{ current_filter }}&sort={{ current_sort }}">Next</a>
            <a href="?page={{ total_pages }}&filter={{ current_filter }}&sort={{ current_sort }}">Last &raquo;</a>
        {% endif %}
    </div>

    <script>
        function updateLabel(basename, newLabel) {
            if(newLabel === 'Unknown') return; // default placeholder
            
            fetch('/update_label', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ basename: basename, new_label: newLabel })
            })
            .then(res => res.json())
            .then(data => {
                if(data.success) {
                    const card = document.getElementById('card-' + basename);
                    const legText = document.getElementById('leg-text-' + basename);
                    
                    // Update text and class
                    legText.innerText = 'Leg: ' + newLabel;
                    legText.className = 'label-' + newLabel;
                    
                    // Trigger flash animation
                    card.classList.remove('updated');
                    void card.offsetWidth; // trigger reflow
                    card.classList.add('updated');
                } else {
                    alert('Lỗi: ' + data.error);
                }
            })
            .catch(err => {
                alert('Yêu cầu thất bại');
            });
        }

        function deleteImage(basename) {
            if(!confirm('Bạn có chắc muốn xóa file ' + basename + ' và toàn bộ nhãn tương ứng (cả ở OCR và Detection) không?')) {
                return;
            }
            fetch('/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ basename: basename })
            })
            .then(res => res.json())
            .then(data => {
                if(data.success) {
                    document.getElementById('card-' + basename).classList.add('deleted');
                    const btn = document.getElementById('card-' + basename).querySelector('.btn-delete');
                    btn.disabled = true;
                    btn.innerText = 'Deleted';
                    btn.style.background = '#6c757d';
                } else {
                    alert('Lỗi: ' + data.error);
                }
            })
            .catch(err => {
                alert('Yêu cầu thất bại');
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    filter_leg = request.args.get('filter', 'all')
    sort_conf = request.args.get('sort', 'none')
    try:
        page = int(request.args.get('page', 1))
    except:
        page = 1

    per_page = 100
    all_images = []

    # Quét tất cả ảnh trong ocr/images/train và ocr/images/val
    for split in ['train', 'val']:
        split_dir = os.path.join(OCR_IMAGES_DIR, split)
        if os.path.exists(split_dir):
            for fname in os.listdir(split_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    basename = fname
                    pred = pred_lookup.get(basename, {})
                    
                    leg = pred.get('legibility', 'Unknown')
                    if filter_leg != 'all' and leg != filter_leg:
                        continue
                        
                    all_images.append({
                        'basename': basename,
                        'rel_path': f"{split}/{fname}",
                        'legibility': leg,
                        'conf': pred.get('conf', 0.0),
                        'route': pred.get('route', 'Unknown'),
                        'quality_score': pred.get('quality_score', 0.0)
                    })

    if sort_conf == 'asc':
        all_images.sort(key=lambda x: x['conf'])
    elif sort_conf == 'desc':
        all_images.sort(key=lambda x: x['conf'], reverse=True)

    total = len(all_images)
    total_pages = (total + per_page - 1) // per_page
    if total_pages == 0:
        total_pages = 1
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_images = all_images[start_idx:end_idx]

    return render_template_string(
        HTML_TEMPLATE, 
        images=page_images, 
        page=page, 
        total=total, 
        total_pages=total_pages,
        current_filter=filter_leg,
        current_sort=sort_conf,
        max=max,
        min=min
    )

@app.route('/image')
def get_image():
    rel_path = request.args.get('path')
    if not rel_path:
        return "Not found", 404
    abs_path = os.path.join(OCR_IMAGES_DIR, rel_path)
    if os.path.exists(abs_path):
        return send_file(abs_path)
    return "Not found", 404

@app.route('/update_label', methods=['POST'])
def update_label():
    global df_train, df_val
    data = request.json
    basename = data.get('basename')
    new_label = data.get('new_label')
    if not basename or not new_label:
        return jsonify({'success': False, 'error': 'Missing parameters'})

    updated = False
    
    # Update df_train
    if not df_train.empty:
        mask = df_train['path'].apply(lambda x: os.path.basename(str(x)) == basename)
        if mask.any():
            df_train.loc[mask, 'predicted_legibility'] = new_label
            df_train.to_csv(CSV_TRAIN, index=False)
            updated = True
            
    # Update df_val if not found in train
    if not updated and not df_val.empty:
        mask = df_val['path'].apply(lambda x: os.path.basename(str(x)) == basename)
        if mask.any():
            df_val.loc[mask, 'predicted_legibility'] = new_label
            df_val.to_csv(CSV_VAL, index=False)
            updated = True

    if updated:
        # Update lookup dictionary so UI reloads correctly
        if basename in pred_lookup:
            pred_lookup[basename]['legibility'] = new_label
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Image not found in CSV predictions'})

@app.route('/delete', methods=['POST'])
def delete_image():
    data = request.json
    basename = data.get('basename')
    if not basename:
        return jsonify({'success': False, 'error': 'No basename provided'})

    txt_basename = os.path.splitext(basename)[0] + '.txt'
    deleted_files = []

    for split in ['train', 'val']:
        paths_to_remove = [
            os.path.join(OCR_IMAGES_DIR, split, basename),
            os.path.join(OCR_LABELS_DIR, split, txt_basename),
            os.path.join(DET_IMAGES_DIR, split, basename),
            os.path.join(DET_LABELS_DIR, split, txt_basename)
        ]
        
        for p in paths_to_remove:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    deleted_files.append(p)
                except Exception as e:
                    print(f"Error removing {p}: {e}")

    return jsonify({'success': True, 'deleted': deleted_files})

if __name__ == '__main__':
    print("Khởi động server tại: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
