import { useEffect, useRef, useState } from "react";

import { Badge, Button, EmptyState, SegmentedControl, Select, cx } from "../ui";
import { OCR_OPTIONS, PREPROCESS_OPTIONS, formatBytes, formatDuration } from "./constants";

const SOURCE_OPTIONS = [{ value: "video", label: "Tải video" }];

export default function SourcePanel({
  onFileSelect,
  preprocessMode,
  onPreprocessModeChange,
  ocrBackend,
  onOcrBackendChange,
  disabled = false,
  compact = false,
}) {
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [source, setSource] = useState("video");
  const [file, setFile] = useState(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [videoDuration, setVideoDuration] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [recording, setRecording] = useState(false);

  useEffect(() => () => stopCamera(), []);

  const pick = (selected) => {
    if (selected) {
      setFile(selected);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(selected));
      setVideoDuration(null);
    } else {
      setFile(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setVideoDuration(null);
    }
  };

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleVideoMeta = (event) => {
    setVideoDuration(event.target.duration);
  };

  const start = async () => {
    if (!file || loading || disabled) return;
    setLoading(true);
    try {
      await onFileSelect(file);
    } finally {
      setLoading(false);
    }
  };

  const stopCamera = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    setRecording(false);
  };

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      setCameraError(err.message || "Không mở được camera.");
    }
  };

  const startRecording = async () => {
    if (!streamRef.current) await startCamera();
    if (!streamRef.current) return;
    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: "video/webm",
    });
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data?.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      setFile(
        new File([blob], `camera-clip-${Date.now()}.webm`, {
          type: "video/webm",
        }),
      );
      setRecording(false);
    };
    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
  };

  return (
    <section className="surface-panel overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Nguồn dữ liệu</p>
          <h2 className="mt-1 text-lg font-bold">
            Chọn video để phân tích
          </h2>
        </div>
        <Badge tone="info">ALPR ưu tiên video</Badge>
      </div>

      <div className="space-y-4 p-4">
        {!compact && (
          <SegmentedControl
            value={source}
            onChange={setSource}
            options={SOURCE_OPTIONS}
            className="w-full justify-start"
          />
        )}

        {source === "video" && !compact && (
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDrag(false);
              pick(event.dataTransfer.files?.[0]);
            }}
            className={cx(
              "cursor-pointer rounded-[var(--radius-panel)] border border-dashed p-6 text-center transition-colors duration-200",
              drag
                ? "border-cyan-300 bg-cyan-300/10"
                : "border-[var(--color-border)] bg-black/10 hover:border-[var(--color-border-strong)] hover:bg-white/5",
            )}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(event) => pick(event.target.files?.[0])}
            />
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-cyan-300/30 bg-cyan-300/10 text-cyan-100">
              <span aria-hidden>↑</span>
            </div>
            <p className="font-semibold">
              Kéo video vào đây hoặc chọn tệp
            </p>
            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
              MP4, WebM, MOV, AVI, MKV.
            </p>
          </div>
        )}

        {source === "camera" && !compact && (
          <div className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black/20 p-3">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="aspect-video w-full rounded-lg bg-black object-contain"
            />
            {cameraError && (
              <div className="mt-3 rounded-lg border border-red-300/30 bg-red-500/10 px-3 py-2 text-sm text-red-100">
                {cameraError}
              </div>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              <Button size="sm" onClick={startCamera}>
                Mở camera
              </Button>
              {!recording ? (
                <Button
                  size="sm"
                  variant="primary"
                  onClick={startRecording}
                >
                  Ghi clip WebM
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="danger"
                  onClick={stopRecording}
                >
                  Dừng ghi
                </Button>
              )}
              <Button size="sm" variant="ghost" onClick={stopCamera}>
                Tắt camera
              </Button>
            </div>
          </div>
        )}

        {source === "image" && !compact && (
          <EmptyState title="Ảnh tĩnh chưa được nối máy chủ">
            Giai đoạn này giữ API hiện có. Khi thêm POST /upload/image,
            thẻ này sẽ dùng cùng kết quả từ bàn kiểm chứng.
          </EmptyState>
        )}

        {file && (
          <div className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
            <div className="relative overflow-hidden rounded-t-[var(--radius-panel)] bg-black">
              <video
                src={previewUrl}
                controls
                className="block max-h-[360px] w-full object-contain"
                onLoadedMetadata={handleVideoMeta}
              />
            </div>
            <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold">
                  {file.name}
                </p>
                <p className="mt-1 text-xs text-[var(--color-text-subtle)]">
                  {formatBytes(file.size)} · {formatDuration(videoDuration)} ·{" "}
                  {file.type || "tệp video"}
                </p>
              </div>
              <Button
                variant="primary"
                loading={loading}
                disabled={disabled || !file}
                onClick={start}
              >
                Chạy nhận dạng
              </Button>
            </div>
          </div>
        )}

        <div className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black/10">
          {compact ? (
            <div className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-[var(--color-text-muted)]">
              Thiết lập mô hình
              <span className="data-font text-xs">đã khóa</span>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setAdvancedOpen((value) => !value)}
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            >
              Thiết lập mô hình
              <span className="data-font text-xs">
                {advancedOpen ? "ẩn" : "mở"}
              </span>
            </button>
          )}
          {advancedOpen && !compact && (
            <div className="grid gap-3 border-t border-[var(--color-border)] p-4 md:grid-cols-2">
              <Select
                label="Tiền xử lý"
                value={preprocessMode}
                onChange={(event) =>
                  onPreprocessModeChange(event.target.value)
                }
                options={PREPROCESS_OPTIONS}
              />
              <Select
                label="Bộ OCR"
                value={ocrBackend}
                onChange={(event) =>
                  onOcrBackendChange(event.target.value)
                }
                options={OCR_OPTIONS}
              />
            </div>
          )}
          {compact && (
            <div className="border-t border-[var(--color-border)] px-4 py-3 text-xs text-[var(--color-text-muted)]">
              {PREPROCESS_OPTIONS.find(
                (option) => option.value === preprocessMode,
              )?.label || preprocessMode}
              {" · "}
              {OCR_OPTIONS.find(
                (option) => option.value === ocrBackend,
              )?.label || ocrBackend}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
