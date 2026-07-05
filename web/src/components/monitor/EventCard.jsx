import { useState } from "react";

import PlateDisplay from "../PlateDisplay";
import { Badge, Button, Progress, cx } from "../ui";
import {
    VEHICLE_LABEL,
    averageConfidence,
    cleanPlateText,
    confidenceTone,
} from "../workbench/constants";
import EventDetail from "./EventDetail";
import { getEventVehicles, getVehicleClusters } from "./eventDisplay";

function fmtTime(iso) {
    if (!iso) return "--:--";
    return new Date(iso).toLocaleTimeString("vi");
}

export default function EventCard({ event }) {
    const [expanded, setExpanded] = useState(false);
    const { id, status, markedAt, windowStartSec, windowEndSec, pct, error } =
        event;
    const vehicleList = getEventVehicles(event);
    const tone =
        status === "completed"
            ? "success"
            : status === "failed"
              ? "danger"
              : "info";
    const statusLabel = getStatusLabel(status);
    const hasVehicles = vehicleList.length > 0;

    return (
        <article className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
            <div className="flex items-start justify-between gap-3">
                <div>
                    <p className="data-font text-xs text-[var(--color-text-subtle)]">
                        #{id.slice(-8)} · {fmtTime(markedAt)}
                    </p>
                    <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                        Đoạn {(windowEndSec - windowStartSec).toFixed(1)}s ·{" "}
                        {vehicleList.length} phương tiện
                    </p>
                </div>
                <Badge tone={tone}>{statusLabel}</Badge>
            </div>

            {(status === "pending" || status === "processing") && (
                <div className="mt-3">
                    <div className="mb-2 flex items-center justify-between text-xs text-[var(--color-text-muted)]">
                        <span>Đang phân tích sự kiện</span>
                        <span className="data-font">
                            {pct ? `${pct}%` : "đang chờ"}
                        </span>
                    </div>
                    <Progress value={pct || 8} />
                </div>
            )}

            {hasVehicles && (
                <div className="mt-3 space-y-3">
                    {vehicleList.map((vehicle) => (
                        <EventResultCard
                            key={
                                vehicle.recognition_id ??
                                vehicle.track_id ??
                                vehicle.id
                            }
                            vehicle={vehicle}
                            onInspect={() => setExpanded(true)}
                        />
                    ))}
                </div>
            )}

            {status === "failed" && (
                <div className="mt-3 rounded-lg border border-red-300/30 bg-red-500/10 px-3 py-2 text-xs text-red-100">
                    {error}
                </div>
            )}

            {hasVehicles && (
                <>
                    <button
                        type="button"
                        onClick={() => setExpanded((value) => !value)}
                        className="mt-3 text-sm font-semibold text-cyan-100 hover:text-cyan-50"
                    >
                        {expanded ? "Ẩn các bộ đệm" : "Xem các bộ đệm"}
                    </button>
                    {expanded && <EventDetail event={event} />}
                </>
            )}
        </article>
    );
}

function EventResultCard({ vehicle, onInspect }) {
    const plate = cleanPlateText(vehicle.plate_text ?? vehicle.plate);
    const conf = averageConfidence(
        vehicle.chars,
        vehicle.plate_text_confidence ?? vehicle.confidence,
    );
    const tone = confidenceTone(conf);
    const frameCount = vehicle.ocr_frames || vehicle.frame_count || 0;
    const identityLabel = formatRecognitionIdentity(vehicle);
    const clusters = getVehicleClusters(vehicle);

    return (
        <article className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                        <p className="plate-font truncate text-lg font-bold tracking-widest text-[var(--color-text)]">
                            {plate ||
                                (vehicle.done
                                    ? "Không thể đọc"
                                    : "Đang nhận dạng")}
                        </p>
                        <Badge tone={vehicle.done ? "success" : "info"}>
                            {vehicle.done ? "Đã xác nhận" : "Đang xử lý"}
                        </Badge>
                    </div>
                    <p className="mt-1 text-xs text-[var(--color-text-subtle)]">
                        {identityLabel} ·{" "}
                        {VEHICLE_LABEL[vehicle.cls] ||
                            vehicle.cls ||
                            "Phương tiện"}{" "}
                        · {frameCount} khung
                    </p>
                </div>
                <div className="data-font text-right text-sm font-bold text-[var(--color-text)]">
                    {conf > 0 ? `${conf}%` : "—"}
                </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2">
                <EvidenceThumb
                    src={vehicle.vehicle_b64 || vehicle.vehicle_image_url}
                    label="Ảnh cắt phương tiện"
                />
                <EvidenceThumb
                    src={vehicle.plate_b64 || vehicle.plate_image_url}
                    label="Ảnh cắt biển số"
                    dark
                />
            </div>

            <div className="mt-3">
                <PlateDisplay chars={vehicle.chars} compact />
            </div>

            <div className="mt-3 flex items-center justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <Progress
                        value={conf}
                        tone={tone === "neutral" ? "info" : tone}
                    />
                </div>
                <Button size="sm" variant="secondary" onClick={onInspect}>
                    Kiểm tra
                </Button>
            </div>

            {clusters.length > 1 && (
                <div className="mt-4 space-y-2 border-t border-[var(--color-border)] pt-3">
                    <p className="text-xs font-semibold text-[var(--color-text-muted)]">
                        Phát hiện nhiều biển số
                    </p>
                    {clusters.map((cluster, idx) => {
                        const clusterConf = averageConfidence(
                            cluster.chars,
                            cluster.plate_text_confidence ?? cluster.confidence,
                        );
                        return (
                            <div
                                key={cluster.cluster_index ?? idx}
                                className="rounded-lg border border-[var(--color-border)] bg-black/15 p-2"
                            >
                                <div className="flex items-center justify-between">
                                    <Badge tone="neutral">Cụm {idx + 1}</Badge>
                                    <span className="text-[10px] text-[var(--color-text-subtle)]">
                                        {cluster.frame_count ||
                                            cluster.ocr_frames ||
                                            0}{" "}
                                        khung ·{" "}
                                        {clusterConf > 0
                                            ? `${clusterConf}%`
                                            : "—"}
                                    </span>
                                </div>
                                <div className="mt-2 grid grid-cols-[auto_1fr] gap-3">
                                    <div className="w-20">
                                        <EvidenceThumb
                                            src={
                                                cluster.plate_b64 ||
                                                cluster.plate_image_url
                                            }
                                            label="Ảnh biển số"
                                            dark
                                        />
                                    </div>
                                    <div className="min-w-0 flex flex-col justify-center">
                                        <p className="plate-font truncate text-sm font-bold tracking-widest text-white">
                                            {cleanPlateText(
                                                cluster.plate_text ??
                                                    cluster.plate,
                                            ) || "—"}
                                        </p>
                                        <div className="mt-1">
                                            <PlateDisplay
                                                chars={cluster.chars}
                                                compact
                                            />
                                        </div>
                                        <div className="mt-2 flex justify-end">
                                            <Button
                                                size="sm"
                                                variant="secondary"
                                                onClick={onInspect}
                                            >
                                                Kiểm tra
                                            </Button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </article>
    );
}

function EvidenceThumb({ src, label, dark = false }) {
    return (
        <div
            className={cx(
                "overflow-hidden rounded-lg border border-[var(--color-border)]",
                dark ? "bg-black" : "bg-black/30",
            )}
        >
            <div className="flex h-24 items-center justify-center">
                {src ? (
                    <img
                        src={imageSrc(src)}
                        alt={label}
                        className="max-h-full max-w-full object-contain"
                    />
                ) : (
                    <span className="text-xs text-[var(--color-text-subtle)]">
                        Không có ảnh
                    </span>
                )}
            </div>
            <p className="border-t border-[var(--color-border)] px-2 py-1 text-[10px] font-semibold text-[var(--color-text-subtle)]">
                {label}
            </p>
        </div>
    );
}

function formatRecognitionIdentity(vehicle) {
    const resultId = vehicle.recognition_id ?? vehicle.track_id ?? vehicle.id;
    const parts = [`Kết quả #${resultId}`];
    if (
        vehicle.vehicle_track_id !== undefined &&
        vehicle.vehicle_track_id !== null
    ) {
        parts.push(`Xe #${vehicle.vehicle_track_id}`);
    }
    if (
        vehicle.plate_track_id !== undefined &&
        vehicle.plate_track_id !== null
    ) {
        parts.push(`Biển số #${vehicle.plate_track_id}`);
    }
    return parts.join(" · ");
}

function imageSrc(value) {
    if (!value) return null;
    if (value.startsWith("http") || value.startsWith("data:")) return value;
    return `data:image/jpeg;base64,${value}`;
}

function getStatusLabel(status) {
    if (status === "completed") return "Hoàn tất";
    if (status === "failed") return "Có lỗi";
    if (status === "processing") return "Đang xử lý";
    return "Đang chờ";
}
