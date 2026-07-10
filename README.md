# ALPR Vietnamese

Hệ thống nhận dạng biển số xe Việt Nam từ video và luồng camera. Backend sử dụng FastAPI, frontend sử dụng React/Vite; dữ liệu được lưu bằng MongoDB và ảnh có thể được lưu trên Supabase Storage.

## Yêu cầu

- Python 3.10+
- Node.js 20+
- FFmpeg
- Docker và Docker Compose nếu chạy bằng container
- GPU NVIDIA được khuyến nghị khi xử lý video

Các tệp trọng số mô hình không nằm trong Git. Trước khi chạy, cần đặt chúng đúng vị trí được khai báo trong `api/.env`.

## Chạy trên máy cá nhân

### Backend

Tại thư mục gốc của dự án:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp api/.env.example api/.env
```

Kiểm tra lại các đường dẫn model trong `api/.env`. Nếu MongoDB và MediaMTX chạy bằng Docker còn backend chạy trực tiếp trên máy, đặt:

```env
MONGODB_URI=mongodb://localhost:27017
MEDIAMTX_API_URL=http://localhost:9997
MEDIAMTX_INTERNAL_RTSP_BASE=rtsp://localhost:8554
MEDIAMTX_PUBLIC_WEBRTC_BASE=http://localhost:8889
```

Khởi động các dịch vụ phụ trợ và backend:

```bash
docker compose up -d mongo mediamtx
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend chạy tại `http://localhost:8000`.

### Frontend

Mở terminal khác:

```bash
cd web
npm ci
npm run dev
```

Truy cập `http://localhost:5173`. Vite đã được cấu hình để chuyển tiếp các yêu cầu API sang backend tại cổng `8000`.

## Chạy bằng Docker Compose

Tạo secret cho phiên đăng nhập trong tệp `.env` ở thư mục gốc:

```bash
printf 'AUTH_SECRET_KEY=%s\n' "$(openssl rand -hex 32)" > .env
docker compose up --build -d
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- MediaMTX WebRTC: `http://localhost:8889`
- MongoDB: `localhost:27017`

Xem log hoặc dừng hệ thống:

```bash
docker compose logs -f api
docker compose down
```

Trọng số model được loại khỏi Docker build context, vì vậy khi triển khai cần mount thư mục model vào container `api` và đặt các biến `VEHICLE_MODEL_PATH`, `PLATE_MODEL_PATH`, `SMALL_LPR_LINE_CTC_CKPT_PATH`, `REID_MODEL_PATH` trỏ tới các đường dẫn trong container.

## Triển khai trên RunPod

### 1. Build và đẩy image backend

```bash
export GH_USER=<github-username>
export IMAGE=ghcr.io/$GH_USER/alpr-vietnamese-backend:runpod-3090

echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GH_USER" --password-stdin
docker buildx build --platform linux/amd64 -f Dockerfile.runpod -t "$IMAGE" --push .
```

### 2. Tạo Pod

Tạo Pod dùng GPU RTX 3090 từ image trên, mount Network Volume tại `/workspace` và mở hai cổng:

- `8000/http`: FastAPI
- `8889/http`: MediaMTX WebRTC

Chép các model vào `/workspace/alpr/models`. Có thể dùng gói `alpr-models-runpod-3090.tar.gz`:

```bash
ssh -p <ssh-port> root@<pod-ip> "mkdir -p /workspace/alpr"
rsync -avP -e "ssh -p <ssh-port>" alpr-models-runpod-3090.tar.gz root@<pod-ip>:/workspace/alpr/
ssh -p <ssh-port> root@<pod-ip> \
  "cd /workspace/alpr && tar -xzf alpr-models-runpod-3090.tar.gz"
```

### 3. Cấu hình biến môi trường của Pod

```env
PORT=8000
WEB_ORIGIN=https://<ten-du-an>.vercel.app
AUTH_COOKIE_SECURE=true
AUTH_SECRET_KEY=<secret-ngau-nhien-dai>
MONGODB_URI=<mongodb-atlas-uri>
MONGODB_DB_NAME=alpr

OCR_BACKEND=smalllpr_line_ctc
VEHICLE_DETECTOR_BACKEND=yolov5
VEHICLE_MODEL_PATH=/workspace/alpr/models/vehicle_object.pt
PLATE_MODEL_PATH=/workspace/alpr/models/plate_obb_best.pt
SMALL_LPR_LINE_CTC_CKPT_PATH=/workspace/alpr/models/small_lpr_line_ctc.ckpt
REID_MODEL_PATH=/workspace/alpr/models/vehicle_reid.onnx
PLATE_QUALITY_ROUTER_MODEL=/workspace/alpr/models/plate_quality_router_best.pt

MEDIAMTX_API_URL=http://127.0.0.1:9997
MEDIAMTX_INTERNAL_RTSP_BASE=rtsp://127.0.0.1:8554
MEDIAMTX_PUBLIC_WEBRTC_BASE=https://<pod-id>-8889.proxy.runpod.net
MONITOR_UPLOAD_DIR=/workspace/alpr/uploads
```

Frontend có thể triển khai từ thư mục `web/` lên Vercel. Thay URL RunPod hiện có trong `web/vercel.json` bằng `https://<pod-id>-8000.proxy.runpod.net`, sau đó đặt `WEB_ORIGIN` của backend đúng bằng tên miền Vercel.

Hướng dẫn RunPod chi tiết hơn nằm tại [deploy/runpod/README.md](deploy/runpod/README.md).
