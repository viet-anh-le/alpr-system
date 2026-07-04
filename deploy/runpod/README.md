# RunPod RTX 3090 Deployment

This repo deploys the full backend on one RunPod Pod:

- FastAPI on port `8000`
- MediaMTX on port `8889`
- MongoDB Atlas and Supabase remain external services
- Model files live on the RunPod network volume at `/workspace/alpr/models`
- The default image is for `OCR_BACKEND=smalllpr_line_ctc`; PARSeq adds extra dependencies and is intentionally not installed in this lean inference image.

## 1. Build and push the image

```bash
export GH_USER=<github-username>
export IMAGE=ghcr.io/$GH_USER/alpr-vietnamese-backend:runpod-3090

echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GH_USER" --password-stdin
docker buildx build --platform linux/amd64 -f Dockerfile.runpod -t "$IMAGE" --push .
```

## 2. Create the Pod

Create an RTX 3090 Pod from the image above.

Expose ports:

- `8000/http`
- `8889/http`
- `22/tcp` if SSH is needed

Mount the network volume at `/workspace`.

## 3. Upload model files

```bash
export RUNPOD_IP=<runpod-public-ip>
export RUNPOD_SSH_PORT=<runpod-ssh-port>

ssh -p "$RUNPOD_SSH_PORT" root@"$RUNPOD_IP" "mkdir -p /workspace/alpr/models /workspace/alpr/uploads"

rsync -avP -e "ssh -p $RUNPOD_SSH_PORT" \
  weights/ root@"$RUNPOD_IP":/workspace/alpr/models/weights/

rsync -avP -e "ssh -p $RUNPOD_SSH_PORT" \
  runs/obb/experiments/detection/lp_detection_obb_merged/weights/ \
  root@"$RUNPOD_IP":/workspace/alpr/models/runs/obb/experiments/detection/lp_detection_obb_merged/weights/

# Optional: only if you want the trained quality router instead of heuristic routing.
rsync -avP -e "ssh -p $RUNPOD_SSH_PORT" \
  runs/classify/legibility_finetuned_vn/weights/ \
  root@"$RUNPOD_IP":/workspace/alpr/models/runs/classify/legibility_finetuned_vn/weights/
```

## 4. Required RunPod env

```bash
PORT=8000
WEB_ORIGIN=https://<vercel-project>.vercel.app
AUTH_COOKIE_SECURE=true
AUTH_SECRET_KEY=<new-long-random-secret>
MONGODB_URI=<mongodb-atlas-uri>
MONGODB_DB_NAME=alpr

SUPABASE_URL=<supabase-url>
SUPABASE_KEY=<supabase-service-role-key>
SUPABASE_STORAGE_HTTP2=false

MAX_UPLOAD_MB=512
OCR_BACKEND=smalllpr_line_ctc

MEDIAMTX_API_URL=http://127.0.0.1:9997
MEDIAMTX_INTERNAL_RTSP_BASE=rtsp://127.0.0.1:8554
MEDIAMTX_PUBLIC_WEBRTC_BASE=https://<pod-id>-8889.proxy.runpod.net
MEDIAMTX_PUBLIC_MJPEG_BASE=
MONITOR_UPLOAD_DIR=/workspace/alpr/uploads

VEHICLE_DETECTOR_BACKEND=yolov5
VEHICLE_MODEL_PATH=/workspace/alpr/models/vehicle_object.pt
PLATE_MODEL_PATH=/workspace/alpr/models/plate_obb_best.pt
SMALL_LPR_LINE_CTC_CKPT_PATH=/workspace/alpr/models/small_lpr_line_ctc.ckpt
REID_MODEL_PATH=/workspace/alpr/models/vehicle_reid.onnx
REID_DEVICE=cpu
PLATE_QUALITY_ROUTER_MODEL=/workspace/alpr/models/plate_quality_router_best.pt
```

If the optional quality-router model is not uploaded, leave `PLATE_QUALITY_ROUTER_MODEL` unset.

## 5. Vercel

Deploy `web/` as the Vercel project root.

Replace every `https://replace-with-runpod-8000.proxy.runpod.net` in `web/vercel.json` with the actual RunPod `8000/http` proxy URL.

Frontend env:

```bash
VITE_SUPABASE_URL=<supabase-url>
VITE_SUPABASE_ANON_KEY=<supabase-anon-key>
```

## 6. Smoke checks

Inside the Pod:

```bash
curl http://127.0.0.1:8000/
nvidia-smi
```

Outside the Pod:

```bash
curl -i https://<pod-id>-8000.proxy.runpod.net/
curl -i https://<vercel-project>.vercel.app/auth/me
```

`/auth/me` should return `401` before login.
