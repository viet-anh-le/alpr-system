# Incident Monitor — Manual E2E Checklist

Run these after `docker compose up -d` and `npm run dev`.

## Upload mode

- [ ] Switch to "Giám sát sự cố" tab.
- [ ] Upload `tests/fixtures/short_clip.mp4`.
- [ ] Video player appears, controls work, no console errors.
- [ ] Click "🚩 Mark Interval" → video pauses, IntervalPicker overlay appears.
- [ ] Drag the two handles to a 5s sub-range.
- [ ] Confirm Δ readout updates in real time.
- [ ] Click "Phân tích" → an IncidentCard appears in pending state.
- [ ] Card transitions to "completed" after analysis.
- [ ] Card shows zero vehicles for the fixture (it has no plates).
- [ ] Mark a SECOND interval — both cards remain visible.
- [ ] Refresh page → cards disappear (in-memory only); incident still in Mongo.

## Live mode (requires a reachable RTSP source)

- [ ] Switch to "Giám sát sự cố" tab → choose "RTSP camera".
- [ ] Paste an RTSP URL (any public test stream or local camera).
- [ ] Click "Kết nối" → LiveViewer renders the WebRTC stream within ~2s.
- [ ] If WebRTC fails, MJPEG fallback banner appears and `<img>` shows frames.
- [ ] Click "🚩 Mark Now" → card appears immediately as pending.
- [ ] Card transitions to completed once analysis finishes.
- [ ] Click "🚩 Mark Now" within 1s of connecting → expect 409 "Buffer warming up".
- [ ] Refresh page → LiveSession is torn down (MediaMTX path removed via `curl http://localhost:9997/v3/paths/list`).

## Error paths

- [ ] Submit interval > 30s → "Interval exceeds 30s max" alert.
- [ ] Submit invalid RTSP URL (http://...) → "URL must be rtsp:// or rtsps://".
- [ ] Stop MediaMTX container mid-live-session → eventually LiveSession reports failure (check server logs).

## Regression on legacy flow

- [ ] Switch back to "Xử lý video" → upload a video → full processing works exactly as before.
