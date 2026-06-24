import os
import re
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

DATASET_DIR = "/home/vietanh/Documents/DATN/ALPR_Vietnamese/data/datasets/ocr"

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALPR Dataset Renamer Grid</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --border: rgba(255, 255, 255, 0.1);
            --primary: #3b82f6;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }
        body {
            margin: 0;
            padding: 2rem;
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            min-height: 100vh;
        }
        .header {
            margin-bottom: 2rem;
            text-align: center;
        }
        h1 {
            margin: 0 0 0.5rem 0;
            font-weight: 600;
        }
        .subtitle {
            color: var(--text-muted);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            border-color: rgba(255,255,255,0.2);
        }
        .image-container {
            width: 100%;
            height: 180px;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-bottom: 1px solid var(--border);
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .card-body {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .meta-info {
            font-size: 0.8rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .input-group {
            display: flex;
            gap: 0.5rem;
        }
        .file-name-input {
            flex: 1;
            background: rgba(0,0,0,0.2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.6rem;
            border-radius: 6px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
            width: 100%;
            box-sizing: border-box;
        }
        .file-name-input:focus {
            border-color: var(--primary);
        }
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: opacity 0.2s;
            white-space: nowrap;
        }
        .btn:hover {
            opacity: 0.9;
        }
        .btn-success {
            background: #10b981;
        }
        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #10b981;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 50;
        }
        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
    </style>
</head>
<body>
    <div class="header">
        <h1>ALPR Dataset Renamer</h1>
        <div class="subtitle" id="count-info">Loading files...</div>
    </div>
    
    <div class="grid" id="file-grid">
        <!-- Cards will be injected here -->
    </div>
    
    <div class="toast" id="toast">Saved successfully!</div>

    <script>
        let currentFiles = [];

        async function loadFiles() {
            const res = await fetch('/api/files');
            currentFiles = await res.json();
            document.getElementById('count-info').textContent = `${currentFiles.length} files found`;
            renderFiles();
        }

        function renderFiles() {
            const grid = document.getElementById('file-grid');
            grid.innerHTML = '';
            
            if(currentFiles.length === 0) {
                grid.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 3rem;">No matching files found.</div>';
                return;
            }

            currentFiles.forEach((file, index) => {
                const card = document.createElement('div');
                card.className = 'card';
                card.id = `card-${index}`;
                
                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-container';
                
                const img = document.createElement('img');
                img.loading = 'lazy';
                img.src = '/image?path=' + encodeURIComponent(file.abs_path);
                img.id = `img-${index}`;
                
                imgContainer.appendChild(img);
                
                const body = document.createElement('div');
                body.className = 'card-body';
                
                const meta = document.createElement('div');
                meta.className = 'meta-info';
                meta.textContent = file.rel_path;
                meta.title = file.rel_path;
                meta.id = `meta-${index}`;
                
                const inputGroup = document.createElement('div');
                inputGroup.className = 'input-group';
                
                const input = document.createElement('input');
                input.className = 'file-name-input';
                input.value = file.name;
                
                const saveBtn = document.createElement('button');
                saveBtn.className = 'btn';
                saveBtn.textContent = 'Save';
                saveBtn.onclick = () => renameFile(index, input.value, saveBtn);
                
                inputGroup.appendChild(input);
                inputGroup.appendChild(saveBtn);
                
                body.appendChild(meta);
                body.appendChild(inputGroup);
                
                card.appendChild(imgContainer);
                card.appendChild(body);
                
                grid.appendChild(card);
            });
        }

        async function renameFile(index, newName, btn) {
            const file = currentFiles[index];
            if(newName === file.name) return;
            
            const originalText = btn.textContent;
            btn.textContent = '...';
            btn.disabled = true;
            
            try {
                const res = await fetch('/api/rename', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        old_path: file.abs_path,
                        new_name: newName
                    })
                });
                
                if(res.ok) {
                    const data = await res.json();
                    file.name = newName;
                    file.abs_path = data.new_path;
                    file.rel_path = data.new_rel_path;
                    
                    document.getElementById(`meta-${index}`).textContent = file.rel_path;
                    document.getElementById(`meta-${index}`).title = file.rel_path;
                    document.getElementById(`img-${index}`).src = '/image?path=' + encodeURIComponent(file.abs_path) + '&t=' + new Date().getTime();
                    
                    btn.classList.add('btn-success');
                    btn.textContent = 'OK';
                    showToast();
                    setTimeout(() => {
                        btn.classList.remove('btn-success');
                        btn.textContent = 'Save';
                        btn.disabled = false;
                    }, 2000);
                } else {
                    alert('Failed to rename file. It might already exist.');
                    btn.textContent = originalText;
                    btn.disabled = false;
                }
            } catch(e) {
                alert('Error: ' + e);
                btn.textContent = originalText;
                btn.disabled = false;
            }
        }

        function showToast() {
            const toast = document.getElementById('toast');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        loadFiles();
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Suppress logs for cleaner terminal

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif path == "/api/files":
            files_data = []
            pattern = re.compile(r"^[A-Z]{2}.*")
            
            for root, _, files in os.walk(DATASET_DIR):
                for filename in files:
                    if pattern.match(filename) and filename.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                        abs_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(abs_path, DATASET_DIR)
                        files_data.append({
                            "name": filename,
                            "abs_path": abs_path,
                            "rel_path": rel_path
                        })
            files_data.sort(key=lambda x: x['name'])
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(files_data).encode("utf-8"))
            
        elif path == "/image":
            qs = urllib.parse.parse_qs(parsed.query)
            img_path = qs.get("path", [""])[0]
            if os.path.exists(img_path) and img_path.startswith(DATASET_DIR):
                self.send_response(200)
                ext = os.path.splitext(img_path)[1].lower()
                mime = "image/jpeg"
                if ext == ".png": mime = "image/png"
                elif ext == ".webp": mime = "image/webp"
                self.send_header("Content-type", mime)
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/rename":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            old_path = data.get("old_path")
            new_name = data.get("new_name")
            
            if old_path and new_name and os.path.exists(old_path) and old_path.startswith(DATASET_DIR):
                dir_name = os.path.dirname(old_path)
                new_path = os.path.join(dir_name, new_name)
                
                # Check if new name doesn't overwrite another file
                if os.path.exists(new_path) and old_path != new_path:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "File already exists"}')
                    return
                
                os.rename(old_path, new_path)
                
                # Also rename corresponding .txt file
                old_base = os.path.splitext(old_path)[0]
                new_base = os.path.splitext(new_path)[0]
                old_txt = old_base + ".txt"
                new_txt = new_base + ".txt"
                if os.path.exists(old_txt) and old_txt != old_path:
                    os.rename(old_txt, new_txt)
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                res = {
                    "status": "success",
                    "new_path": new_path,
                    "new_rel_path": os.path.relpath(new_path, DATASET_DIR)
                }
                self.wfile.write(json.dumps(res).encode("utf-8"))
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid request"}')

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting Dataset Renamer Grid server on http://localhost:{port} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run()
