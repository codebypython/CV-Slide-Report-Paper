"""Interactive Web Dashboard for Pediatric Bone Age Assessment (Pipeline 03b).

Zero-dependency web server built with Python standard library (http.server).
Runs locally at http://localhost:8501
"""

import os
import sys
import json
import base64
import io
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
from src.inference.pipeline_03b import BoneAgeInferencePipeline

# Global pipeline instance
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        print("[INIT] Khoi tao Pipeline 03b...")
        PIPELINE = BoneAgeInferencePipeline()
    return PIPELINE


def img_to_base64(img_rgb: np.ndarray) -> str:
    """Converts numpy RGB image to Base64 PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("utf-8")


def generate_synthetic_xray() -> np.ndarray:
    """Generates synthetic hand radiograph for quick instant testing."""
    synth = np.ones((512, 512, 3), dtype=np.uint8) * 18
    cv2.ellipse(synth, (256, 380), (105, 115), 0, 0, 360, (185, 185, 185), -1)
    fingers = [
        ((168, 238), (14, 62), -26),
        ((214, 168), (15, 82), -8),
        ((256, 148), (16, 92), 0),
        ((298, 168), (15, 82), 8),
        ((342, 208), (14, 66), 19)
    ]
    for center, axes, ang in fingers:
        cv2.ellipse(synth, center, axes, ang, 0, 360, (195, 195, 195), -1)
    return cv2.GaussianBlur(synth, (21, 21), 0)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pediatric Bone Age SOTA Assessment (Pipeline 03b)</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 30, 49, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.25);
      --accent: #10b981;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --gold: #f59e0b;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }
    
    .header {
      max-width: 1280px; margin: 0 auto 24px auto;
      display: flex; justify-content: space-between; align-items: center;
      padding-bottom: 16px; border-bottom: 1px solid var(--card-border);
    }
    .header h1 { font-size: 24px; font-weight: 800; display: flex; align-items: center; gap: 10px; }
    .badge-sota {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #000; font-weight: 700; font-size: 11px; padding: 4px 10px;
      border-radius: 20px; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .subtitle { color: var(--text-muted); font-size: 13px; margin-top: 4px; }

    .main-grid {
      max-width: 1280px; margin: 0 auto;
      display: grid; grid-template-columns: 420px 1fr; gap: 24px;
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px; padding: 24px;
      backdrop-filter: blur(16px); box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    .section-title { font-size: 14px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 14px; letter-spacing: 0.5px; }

    /* Drop Zone */
    .dropzone {
      border: 2px dashed rgba(59, 130, 246, 0.4);
      border-radius: 12px; padding: 32px 16px; text-align: center;
      cursor: pointer; transition: all 0.2s ease; background: rgba(59, 130, 246, 0.04);
      margin-bottom: 16px;
    }
    .dropzone:hover { border-color: var(--primary); background: rgba(59, 130, 246, 0.08); }
    .dropzone img { max-width: 100%; max-height: 200px; border-radius: 8px; margin-top: 10px; display: none; }

    /* Gender Toggle */
    .gender-group { display: flex; gap: 12px; margin-bottom: 20px; }
    .gender-btn {
      flex: 1; padding: 12px; border-radius: 10px;
      border: 1px solid var(--card-border); background: rgba(255,255,255,0.03);
      color: var(--text); font-weight: 600; font-size: 14px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 8px;
      transition: all 0.2s ease;
    }
    .gender-btn.active {
      border-color: var(--primary); background: var(--primary); color: white;
      box-shadow: 0 0 16px var(--primary-glow);
    }

    .btn-action {
      width: 100%; padding: 14px; border-radius: 12px;
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      color: white; font-weight: 700; font-size: 15px; border: none;
      cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 16px var(--primary-glow);
      display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn-action:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }
    
    .btn-sample {
      width: 100%; padding: 10px; border-radius: 10px; margin-top: 10px;
      background: rgba(255,255,255,0.05); color: var(--text-muted);
      border: 1px solid var(--card-border); font-size: 13px; font-weight: 600;
      cursor: pointer; transition: all 0.2s ease;
    }
    .btn-sample:hover { background: rgba(255,255,255,0.08); color: var(--text); }

    /* Results Panel */
    .result-hero {
      display: flex; justify-content: space-between; align-items: center;
      background: rgba(255,255,255,0.03); border: 1px solid var(--card-border);
      border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
    }
    .hero-number { font-size: 48px; font-weight: 800; color: #60a5fa; line-height: 1; }
    .hero-unit { font-size: 14px; font-weight: 700; color: var(--text-muted); margin-top: 6px; }

    .stage-card {
      padding: 16px; border-radius: 12px; margin-bottom: 20px;
      border-left: 4px solid var(--primary); background: rgba(59, 130, 246, 0.08);
    }
    .stage-title { font-weight: 700; font-size: 15px; margin-bottom: 4px; }
    .stage-desc { font-size: 13px; color: var(--text-muted); line-height: 1.5; }

    /* Model Breakdown */
    .breakdown-box {
      background: rgba(0,0,0,0.25); border-radius: 12px;
      padding: 16px; margin-bottom: 20px; border: 1px solid var(--card-border);
    }
    .breakdown-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }
    .breakdown-row:last-child { margin-bottom: 0; }
    .breakdown-val { font-weight: 700; color: #93c5fd; }

    /* Tabs Visualizer */
    .tabs-nav { display: flex; gap: 8px; margin-bottom: 14px; border-bottom: 1px solid var(--card-border); padding-bottom: 10px; }
    .tab-btn {
      padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 600;
      color: var(--text-muted); background: transparent; border: none; cursor: pointer;
    }
    .tab-btn.active { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }

    .visualizer-container {
      background: #000; border-radius: 12px; overflow: hidden;
      display: flex; justify-content: center; align-items: center; min-height: 380px;
      border: 1px solid var(--card-border);
    }
    .visualizer-container img { max-width: 100%; max-height: 440px; object-fit: contain; }

    .spinner { display: none; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 0.8s ease-in-out infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>🦴 Pediatric Bone Age Assessment Studio <span class="badge-sota">SOTA Pipeline 03b</span></h1>
      <p class="subtitle">Harmonizing Whole Hand Macro (02d) & Anatomical Carpal Zoom (02f) via Constrained Ridge Stacking</p>
    </div>
    <div style="text-align: right;">
      <span style="font-size: 12px; color: #10b981; font-weight: 700;">● SẴN SÀNG SUY LUẬN</span>
      <p style="font-size: 11px; color: var(--text-muted);">Target Venue: IEEE TMI</p>
    </div>
  </div>

  <div class="main-grid">
    <!-- CỘT TRÁI: ĐIỀU KHIỂN & TẢI ẢNH -->
    <div class="card">
      <div class="section-title">1. Tải Lên Ảnh X-Quang Bàn Tay</div>
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        <div id="dropText">
          <div style="font-size: 32px; margin-bottom: 8px;">📤</div>
          <p style="font-weight: 600; font-size: 14px;">Kéo thả ảnh hoặc Bấm để chọn tệp</p>
          <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Hỗ trợ ảnh PNG, JPG, BMP (Khuyên dùng chuẩn PA Hand)</p>
        </div>
        <img id="previewImg" alt="Preview">
      </div>

      <button class="btn-sample" onclick="loadSampleImage()">✨ Nạp Ảnh Mẫu Thử Nghiệm Ngay</button>

      <div class="section-title" style="margin-top: 24px;">2. Giới Tính Bệnh Nhi</div>
      <div class="gender-group">
        <button class="gender-btn active" id="btnMale" onclick="selectGender(1)">👦 Bé Trai (Male)</button>
        <button class="gender-btn" id="btnFemale" onclick="selectGender(0)">👧 Bé Gái (Female)</button>
      </div>

      <button class="btn-action" id="btnRun" onclick="runInference()">
        <span class="spinner" id="spinner"></span>
        <span id="btnRunText">🚀 PHÂN TÍCH TUỔI XƯƠNG (03b SOTA)</span>
      </button>
    </div>

    <!-- CỘT PHẢI: KẾT QUẢ & TRỰC QUAN HÓA -->
    <div class="card">
      <div class="section-title">3. Kết Quả Chẩn Đoán Tuổi Xương & Lâm Sàng</div>

      <div class="result-hero">
        <div>
          <div class="hero-number" id="resAgeMonths">--.-</div>
          <div class="hero-unit">THÁNG TUỔI XƯƠNG DỰ ĐOÁN</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 28px; font-weight: 800; color: #a7f3d0;" id="resAgeYears">--.-</div>
          <div class="hero-unit">QUY ĐỔI NĂM TUỔI</div>
        </div>
      </div>

      <div class="stage-card" id="stageCard">
        <div class="stage-title" id="resStageTitle">Đang chờ phân tích...</div>
        <div class="stage-desc" id="resStageDesc">Hãy chọn ảnh X-quang và bấm nút Phân Tích Tuổi Xương để hiển thị đánh giá lâm sàng.</div>
      </div>

      <div class="breakdown-box">
        <div style="font-size: 12px; font-weight: 700; color: var(--text-muted); margin-bottom: 10px; text-transform: uppercase;">
          Đóng Góp Trọng Số Của Từng Nhánh (Stacking Architecture):
        </div>
        <div class="breakdown-row">
          <span>• Nhánh 02d (Toàn Cảnh Bàn Tay Macro):</span>
          <span class="breakdown-val" id="res02d">--.- tháng (Trọng số: 42.0%)</span>
        </div>
        <div class="breakdown-row">
          <span>• Nhánh 02f (Vi Phẫu Cổ Tay Carpal Zoom):</span>
          <span class="breakdown-val" id="res02f">--.- tháng (Trọng số: 58.0%)</span>
        </div>
      </div>

      <div class="section-title" style="margin-top: 20px;">4. Trực Quan Hóa Vùng Chú Ý Giải Phẫu (Explainability)</div>
      <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('tab_hand')">🔍 Vùng Cắt Cổ Tay (02f ROI)</button>
        <button class="tab-btn" onclick="switchTab('tab_crop')">🔬 Phóng To Vi Phẫu (256x256)</button>
        <button class="tab-btn" onclick="switchTab('tab_heatmap')">🔥 Bản Đồ Nhiệt Sụn TW3</button>
      </div>

      <div class="visualizer-container">
        <img id="visImage" src="" alt="Trực quan hóa" style="display: none;">
        <div id="visPlaceholder" style="color: var(--text-muted); font-size: 13px;">Chưa có ảnh chẩn đoán</div>
      </div>
    </div>
  </div>

  <script>
    let currentGender = 1;
    let currentBase64 = null;
    let inferenceData = null;
    let activeTab = 'tab_hand';

    function selectGender(val) {
      currentGender = val;
      document.getElementById('btnMale').className = val === 1 ? 'gender-btn active' : 'gender-btn';
      document.getElementById('btnFemale').className = val === 0 ? 'gender-btn active' : 'gender-btn';
    }

    function handleFileSelect(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        currentBase64 = e.target.result;
        showPreview(currentBase64);
      };
      reader.readAsDataURL(file);
    }

    function showPreview(b64) {
      const img = document.getElementById('previewImg');
      img.src = b64;
      img.style.display = 'block';
      document.getElementById('dropText').style.display = 'none';
    }

    function loadSampleImage() {
      fetch('/api/sample')
        .then(res => res.json())
        .then(data => {
          currentBase64 = 'data:image/png;base64,' + data.base64;
          showPreview(currentBase64);
        });
    }

    function switchTab(tabName) {
      activeTab = tabName;
      document.querySelectorAll('.tab-btn').forEach((b, idx) => {
        b.className = ((idx === 0 && tabName === 'tab_hand') ||
                       (idx === 1 && tabName === 'tab_crop') ||
                       (idx === 2 && tabName === 'tab_heatmap')) ? 'tab-btn active' : 'tab-btn';
      });
      updateVisualizerImage();
    }

    function updateVisualizerImage() {
      if (!inferenceData) return;
      const visImg = document.getElementById('visImage');
      const placeholder = document.getElementById('visPlaceholder');
      let src = null;
      if (activeTab === 'tab_hand') src = inferenceData.images.whole_hand_annotated;
      else if (activeTab === 'tab_crop') src = inferenceData.images.carpal_zoom_crop;
      else if (activeTab === 'tab_heatmap') src = inferenceData.images.attention_heatmap;

      if (src) {
        visImg.src = 'data:image/png;base64,' + src;
        visImg.style.display = 'block';
        placeholder.style.display = 'none';
      }
    }

    function runInference() {
      if (!currentBase64) {
        loadSampleImage();
        setTimeout(runInference, 300);
        return;
      }

      const spinner = document.getElementById('spinner');
      const btnText = document.getElementById('btnRunText');
      spinner.style.display = 'inline-block';
      btnText.innerText = 'Đang tính toán tối ưu Stacking...';

      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: currentBase64.split(',')[1] || currentBase64,
          male: currentGender
        })
      })
      .then(res => res.json())
      .then(data => {
        inferenceData = data;
        document.getElementById('resAgeMonths').innerText = data.predicted_bone_age_months.toFixed(1);
        document.getElementById('resAgeYears').innerText = '(~ ' + data.predicted_bone_age_years.toFixed(1) + ' tuổi)';

        const stageCard = document.getElementById('stageCard');
        document.getElementById('resStageTitle').innerText = 'Giai Đoạn: ' + data.pubertal_stage;
        document.getElementById('resStageDesc').innerText = data.pubertal_description;
        stageCard.style.borderLeftColor = data.pubertal_color;

        const mb = data.model_breakdown;
        document.getElementById('res02d').innerText = mb.pred_02d_months.toFixed(1) + ' tháng (Trọng số: ' + (mb.weight_02d * 100).toFixed(1) + '%)';
        document.getElementById('res02f').innerText = mb.pred_02f_months.toFixed(1) + ' tháng (Trọng số: ' + (mb.weight_02f * 100).toFixed(1) + '%)';

        updateVisualizerImage();
      })
      .catch(err => {
        alert('Lỗi khi phân tích: ' + err);
      })
      .finally(() => {
        spinner.style.display = 'none';
        btnText.innerText = '🚀 PHÂN TÍCH TUỔI XƯƠNG (03b SOTA)';
      });
    }
  </script>
</body>
</html>
"""


class BoneAgeServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/sample":
            synth_rgb = generate_synthetic_xray()
            b64_str = img_to_base64(synth_rgb)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"base64": b64_str}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode("utf-8"))

            b64_img = req.get("image_b64")
            is_male = int(req.get("male", 1))

            img_bytes = base64.b64decode(b64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            pipeline = get_pipeline()
            res = pipeline.predict(img_rgb, is_male=is_male)

            # Convert images to base64 for web payload
            web_response = {
                "predicted_bone_age_months": res["predicted_bone_age_months"],
                "predicted_bone_age_years": res["predicted_bone_age_years"],
                "gender": res["gender"],
                "pubertal_stage": res["pubertal_stage"],
                "pubertal_description": res["pubertal_description"],
                "pubertal_color": res["pubertal_color"],
                "model_breakdown": res["model_breakdown"],
                "images": {
                    "whole_hand_original": img_to_base64(res["images"]["whole_hand_original"]),
                    "whole_hand_annotated": img_to_base64(res["images"]["whole_hand_annotated"]),
                    "carpal_zoom_crop": img_to_base64(res["images"]["carpal_zoom_crop"]),
                    "attention_heatmap": img_to_base64(res["images"]["attention_heatmap"])
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(web_response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8501):
    server_address = ("", port)
    httpd = HTTPServer(server_address, BoneAgeServerHandler)
    print("=" * 70)
    print(f"[START] Ung dung Web Demo Tuoi Xuong 03b da san sang!")
    print(f"[URL]   Mo trinh duyet tai: http://localhost:{port}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Da dung may chu web demo.")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = 8501
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port_arg = int(sys.argv[1])
    start_server(port_arg)
