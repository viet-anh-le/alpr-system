import { Badge, EmptyState, Progress } from "../ui";

function statusCopy(status) {
    if (status === "done") return ["Hoàn tất", "success"];
    if (status === "error") return ["Có lỗi", "danger"];
    if (status === "uploading") return ["Đang tải lên", "info"];
    if (status === "processing") return ["Đang phân tích", "info"];
    return ["Sẵn sàng", "neutral"];
}

export default function MediaStage({ previewFrame, videoUrl, progress, status }) {
    const isDone = status === "done";
    const isError = status === "error";
    const hasPreview = !!previewFrame?.b64;
    const [label, tone] = statusCopy(status);

    return (
        <section className="surface-panel overflow-hidden">
            <div className="panel-header">
                <div>
                    <p className="section-label">Chứng cứ hình ảnh</p>
                </div>
                <Badge tone={tone}>{label}</Badge>
            </div>

            <div className="p-4">
                <div className="relative overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black scanline-bg">
                    {hasPreview && !isDone && (
                        <PreviewFrame frame={previewFrame} />
                    )}

                    {(isDone || !hasPreview) && videoUrl && (
                        <video
                            key={videoUrl}
                            src={videoUrl}
                            controls
                            muted
                            autoPlay={!isDone}
                            loop={isDone}
                            className="block max-h-[68vh] w-full object-contain"
                        />
                    )}

                    {!videoUrl && !hasPreview && (
                        <div className="p-4">
                            <EmptyState title="Chưa có nguồn phân tích">
                                Chọn video tải lên hoặc ghi clip camera để xem
                                khung hình phân tích và tiến trình OCR tại đây.
                            </EmptyState>
                        </div>
                    )}

                    {videoUrl && !isDone && (
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/70 to-transparent px-4 pb-4 pt-16">
                            <div className="mb-2 flex items-center justify-between gap-3 text-sm text-white">
                                <div className="flex items-center gap-2">
                                    {!isDone && !isError && (
                                        <span className="h-2 w-2 rounded-full bg-cyan-300" />
                                    )}
                                    <span className="font-semibold">
                                        {label}
                                    </span>
                                </div>
                                <span className="data-font font-bold">
                                    {Math.round(progress.pct || 0)}%
                                </span>
                            </div>
                            <Progress
                                value={progress.pct || 0}
                                tone={
                                    isDone
                                        ? "success"
                                        : isError
                                          ? "danger"
                                          : "info"
                                }
                            />
                            {progress.frame > 0 && (
                                <p className="mt-2 data-font text-xs text-white/70">
                                    Khung {progress.frame.toLocaleString("vi")}{" "}
                                    / {progress.total.toLocaleString("vi")}
                                </p>
                            )}
                        </div>
                    )}
                </div>
                {videoUrl && isDone && (
                    <div className="mt-3 rounded-[var(--radius-control)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-4 py-3">
                        <div className="mb-2 flex items-center justify-between gap-3 text-sm">
                            <span className="font-semibold">
                                {label} · video gốc đã sẵn sàng để kiểm tra
                            </span>
                            <span className="data-font font-bold">
                                {Math.round(progress.pct || 0)}%
                            </span>
                        </div>
                        <Progress value={progress.pct || 100} tone="success" />
                    </div>
                )}
            </div>
        </section>
    );
}

function PreviewFrame({ frame }) {
    const width = Number(frame.image_width) || 1;
    const height = Number(frame.image_height) || 1;
    const boxes = Array.isArray(frame.boxes) ? frame.boxes : [];

    return (
        <div className="relative">
            <img
                src={`data:image/jpeg;base64,${frame.b64}`}
                alt="Khung hình đang phân tích"
                className="block max-h-[68vh] w-full object-contain"
                draggable={false}
            />
            <svg
                className="pointer-events-none absolute inset-0 h-full w-full"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
                aria-hidden="true"
            >
                {boxes.map((box, index) => (
                    <OverlayBox
                        key={`${box.id ?? "box"}-${index}`}
                        box={box}
                        imageWidth={width}
                    />
                ))}
            </svg>
        </div>
    );
}

function OverlayBox({ box, imageWidth }) {
    const [x1, y1, x2, y2] = Array.isArray(box.box) ? box.box.map(Number) : [0, 0, 0, 0];
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    const color = boxColor(box.state);
    const label = box.label || `${box.cls || "vehicle"} #${box.id ?? ""}`.trim();
    const labelX = Math.max(0, x1);
    const labelY = Math.max(0, y1 - 24);
    const labelWidth = Math.min(
        Math.max(64, label.length * 8 + 12),
        Math.max(64, imageWidth - labelX),
    );
    const compressedTextWidth = Math.max(1, labelWidth - 12);
    const shouldCompressLabel = label.length * 8 > compressedTextWidth;

    return (
        <g>
            <rect
                x={x1}
                y={y1}
                width={width}
                height={height}
                fill="none"
                stroke={color}
                strokeWidth={box.state === "active" ? 3 : 2}
                vectorEffect="non-scaling-stroke"
            />
            {label && (
                <g>
                    <rect
                        x={labelX}
                        y={labelY}
                        width={labelWidth}
                        height="20"
                        rx="4"
                        fill={color}
                        opacity="0.92"
                    />
                    <text
                        x={labelX + 6}
                        y={labelY + 14}
                        fill="#071018"
                        fontSize="13"
                        fontWeight="700"
                        fontFamily="var(--font-data)"
                        textLength={shouldCompressLabel ? compressedTextWidth : undefined}
                        lengthAdjust="spacingAndGlyphs"
                    >
                        {label}
                    </text>
                </g>
            )}
        </g>
    );
}

function boxColor(state) {
    if (state === "active") return "#ffd200";
    if (state === "done") return "#00dc3c";
    return "#b4b4b4";
}
