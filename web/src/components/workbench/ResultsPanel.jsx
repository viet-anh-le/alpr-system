import { useMemo, useState } from "react";

import ImageModal from "../ImageModal";
import PlateDisplay from "../PlateDisplay";
import TrackBufferModal from "../TrackBufferModal";
import { Badge, Button, Drawer, EmptyState, Progress, cx } from "../ui";
import {
    VEHICLE_LABEL,
    averageConfidence,
    cleanPlateText,
    confidenceTone,
} from "./constants";

export default function ResultsPanel({
    vehicles,
    rejectedVehicles,
    totalDone,
    jobId,
    status,
}) {
    const [selected, setSelected] = useState(null);
    const [activeTab, setActiveTab] = useState("confirmed");
    const confirmed = useMemo(() => [...vehicles].reverse(), [vehicles]);
    const rejected = useMemo(
        () => [...rejectedVehicles].reverse(),
        [rejectedVehicles],
    );
    const total = confirmed.length + rejected.length;
    const activeList = activeTab === "rejected" ? rejected : confirmed;
    const activeEmptyTitle =
        activeTab === "rejected"
            ? "Chưa có biển số bị loại"
            : "Chưa có kết quả biển số hợp lệ";
    const activeEmptyCopy =
        activeTab === "rejected"
            ? "Các biển số sai mẫu định dạng, độ tin cậy thấp hoặc không đủ khung hình sẽ xuất hiện trong thẻ này."
            : "Khi hệ thống xác nhận biển số hợp lệ, kết quả và độ tin cậy sẽ xuất hiện trong thẻ này.";

    return (
        <>
            <section className="surface-panel recognition-panel flex flex-col">
                <div className="panel-header">
                    <div>
                        <p className="section-label">Kiểm chứng nhận dạng</p>
                        <h2 className="mt-1 text-lg font-bold">
                            {activeTab === "rejected"
                                ? "Biển số bị loại"
                                : "Biển số đã nhận dạng"}
                        </h2>
                    </div>
                    <Badge
                        tone={
                            status === "done"
                                ? "success"
                                : total > 0
                                  ? "info"
                                  : "neutral"
                        }
                    >
                        {total} biển số
                    </Badge>
                </div>

                <div className="grid grid-cols-3 gap-px border-b border-[var(--color-border)] bg-[var(--color-border)]">
                    <Metric
                        label="Hợp lệ"
                        value={confirmed.length}
                        tone="success"
                        active={activeTab === "confirmed"}
                        onClick={() => setActiveTab("confirmed")}
                    />
                    <Metric
                        label="Đã xác nhận"
                        value={totalDone}
                        tone="info"
                        active={activeTab === "confirmed"}
                        onClick={() => setActiveTab("confirmed")}
                    />
                    <Metric
                        label="Bị loại"
                        value={rejected.length}
                        tone="warning"
                        active={activeTab === "rejected"}
                        onClick={() => setActiveTab("rejected")}
                    />
                </div>

                <div className="flex items-center justify-between gap-3 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2">
                    <p className="text-xs font-semibold text-[var(--color-text-muted)]">
                        {activeTab === "rejected"
                            ? `${rejected.length} biển số bị loại cần kiểm tra riêng`
                            : `${confirmed.length} biển số đã xác nhận trong phiên`}
                    </p>
                    {activeTab === "rejected" && rejected.length > 0 && (
                        <Badge tone="warning">
                            Sai định dạng / độ tin cậy thấp
                        </Badge>
                    )}
                </div>

                <div className="recognition-panel-body p-3">
                    {total === 0 ? (
                        <EmptyState title="Chưa có kết quả biển số">
                            Khi hệ thống tìm thấy phương tiện và biển số, kết
                            quả OCR cùng độ tin cậy sẽ xuất hiện tại đây.
                        </EmptyState>
                    ) : activeList.length === 0 ? (
                        <EmptyState title={activeEmptyTitle}>
                            {activeEmptyCopy}
                        </EmptyState>
                    ) : (
                        <div className="space-y-3">
                            {activeList.map((vehicle) => (
                                <ResultCard
                                    key={`${activeTab}-${vehicle.id}`}
                                    vehicle={vehicle}
                                    jobId={jobId}
                                    rejected={activeTab === "rejected"}
                                    onInspect={(targetVehicle) =>
                                        setSelected({
                                            vehicle: targetVehicle,
                                            rejected: activeTab === "rejected",
                                        })
                                    }
                                />
                            ))}
                        </div>
                    )}
                </div>
            </section>

            <EvidenceDrawer
                item={selected}
                jobId={jobId}
                onClose={() => setSelected(null)}
            />
        </>
    );
}

function Metric({ label, value, tone, active = false, onClick }) {
    const color = {
        success: "text-emerald-100",
        info: "text-cyan-100",
        warning: "text-amber-100",
    }[tone];
    return (
        <button
            type="button"
            onClick={onClick}
            className={cx(
                "bg-[var(--color-bg-elevated)] px-3 py-3 text-center transition-colors duration-200 hover:bg-white/5",
                active && "bg-white/8 ring-1 ring-inset ring-cyan-300/35",
            )}
        >
            <div className={cx("data-font text-xl font-bold", color)}>
                {value}
            </div>
            <div className="mt-1 text-[11px] font-semibold text-[var(--color-text-subtle)]">
                {label}
            </div>
        </button>
    );
}

function ResultCard({ vehicle, rejected = false, onInspect }) {
    const plate = cleanPlateText(vehicle.plate);
    const conf = averageConfidence(vehicle.chars, vehicle.confidence);
    const tone = rejected ? "warning" : confidenceTone(conf);
    const frameCount = vehicle.ocr_frames || 0;
    const identityLabel = formatRecognitionIdentity(vehicle);

    return (
        <article className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                        <p className="plate-font truncate text-lg font-bold tracking-widest text-[var(--color-text)]">
                            {plate ||
                                (rejected || vehicle.done
                                    ? "Không thể đọc"
                                    : "Đang nhận dạng")}
                        </p>
                        <Badge
                            tone={
                                rejected
                                    ? "warning"
                                    : vehicle.done
                                      ? "success"
                                      : "info"
                            }
                        >
                            {rejected
                                ? "Bị loại"
                                : vehicle.done
                                  ? "Đã xác nhận"
                                  : "Đang xử lý"}
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
                    src={vehicle.vehicle_b64}
                    label="Ảnh cắt phương tiện"
                />
                <EvidenceThumb
                    src={vehicle.plate_b64}
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
                <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => onInspect(vehicle)}
                >
                    Kiểm tra
                </Button>
            </div>

            {vehicle.clusters && vehicle.clusters.length > 1 && (
                <div className="mt-4 space-y-2 border-t border-[var(--color-border)] pt-3">
                    <p className="text-xs font-semibold text-[var(--color-text-muted)]">
                        Phát hiện nhiều biển số
                    </p>
                    {vehicle.clusters.map((cluster, idx) => {
                        const clusterConf = Math.round(
                            (cluster.confidence || 0) * 100,
                        );
                        return (
                            <div
                                key={idx}
                                className="rounded-lg border border-[var(--color-border)] bg-black/15 p-2"
                            >
                                <div className="flex items-center justify-between">
                                    <Badge tone="neutral">Cụm {idx + 1}</Badge>
                                    <span className="text-[10px] text-[var(--color-text-subtle)]">
                                        {cluster.frame_count} khung ·{" "}
                                        {clusterConf}%
                                    </span>
                                </div>
                                <div className="mt-2 grid grid-cols-[auto_1fr] gap-3">
                                    <div className="w-20">
                                        <EvidenceThumb
                                            src={cluster.plate_b64}
                                            label=""
                                            dark
                                        />
                                    </div>
                                    <div className="min-w-0 flex flex-col justify-center">
                                        <p className="plate-font truncate text-sm font-bold tracking-widest text-white">
                                            {cleanPlateText(cluster.plate) ||
                                                "—"}
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
                                                onClick={() =>
                                                    onInspect({
                                                        ...vehicle,
                                                        ...cluster,
                                                    })
                                                }
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
                        src={`data:image/jpeg;base64,${src}`}
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

function EvidenceDrawer({ item, jobId, onClose }) {
    const [zoomed, setZoomed] = useState(null);
    const [showBuffer, setShowBuffer] = useState(false);
    const [copied, setCopied] = useState(false);
    const vehicle = item?.vehicle;
    const plate = cleanPlateText(vehicle?.plate);
    const conf = averageConfidence(vehicle?.chars, vehicle?.confidence);
    const identityLabel = vehicle ? formatRecognitionIdentity(vehicle) : "";

    const copy = async () => {
        if (!plate) return;
        await navigator.clipboard.writeText(plate);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
    };

    return (
        <>
            <Drawer
                open={!!vehicle}
                onClose={onClose}
                title={
                    plate ||
                    `Kết quả #${vehicle?.recognition_id ?? vehicle?.id ?? ""}`
                }
                description={
                    vehicle
                        ? `${identityLabel} · ${VEHICLE_LABEL[vehicle.cls] || vehicle.cls || "Phương tiện"} · ${conf || 0}% độ tin cậy`
                        : ""
                }
                className="max-w-3xl"
            >
                {vehicle && (
                    <div className="space-y-4 p-4">
                        <div className="grid gap-3 sm:grid-cols-2">
                            <button
                                type="button"
                                onClick={() =>
                                    vehicle.vehicle_b64 &&
                                    setZoomed({
                                        src: `data:image/jpeg;base64,${vehicle.vehicle_b64}`,
                                        alt: identityLabel,
                                    })
                                }
                                className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black/20 p-2 text-left"
                            >
                                <EvidenceImage
                                    src={vehicle.vehicle_b64}
                                    label="Ảnh phương tiện đối chiếu"
                                />
                            </button>
                            <button
                                type="button"
                                onClick={() =>
                                    vehicle.plate_b64 &&
                                    setZoomed({
                                        src: `data:image/jpeg;base64,${vehicle.plate_b64}`,
                                        alt: `Biển số ${plate}`,
                                    })
                                }
                                className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black p-2 text-left"
                            >
                                <EvidenceImage
                                    src={vehicle.plate_b64}
                                    label="Ảnh biển số đối chiếu"
                                />
                            </button>
                        </div>

                        <div className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
                            <p className="section-label">
                                Độ tin cậy OCR theo ký tự
                            </p>
                            <div className="mt-3">
                                <PlateDisplay chars={vehicle.chars} />
                            </div>
                            <div className="mt-4 flex flex-wrap gap-2">
                                <Button
                                    size="sm"
                                    variant="primary"
                                    disabled={!plate}
                                    onClick={copy}
                                >
                                    {copied
                                        ? "Đã sao chép"
                                        : "Sao chép biển số"}
                                </Button>
                                {jobId && (
                                    <Button
                                        size="sm"
                                        onClick={() => setShowBuffer(true)}
                                    >
                                        Xem bộ đệm theo vết
                                    </Button>
                                )}
                            </div>
                        </div>

                        {vehicle.vote_summary && (
                            <div className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
                                <p className="section-label">
                                    Tổng hợp phiếu OCR
                                </p>
                                <div className="mt-3 flex flex-wrap gap-2">
                                    {Object.entries(vehicle.vote_summary).map(
                                        ([text, count]) => (
                                            <Badge key={text} tone="neutral">
                                                <span className="plate-font">
                                                    {cleanPlateText(text)}
                                                </span>
                                                <span className="text-[var(--color-text-subtle)]">
                                                    ×{count}
                                                </span>
                                            </Badge>
                                        ),
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </Drawer>

            {zoomed && (
                <ImageModal
                    src={zoomed.src}
                    alt={zoomed.alt}
                    onClose={() => setZoomed(null)}
                />
            )}
            {showBuffer && vehicle && jobId && (
                <TrackBufferModal
                    vehicle={vehicle}
                    jobId={jobId}
                    onClose={() => setShowBuffer(false)}
                />
            )}
        </>
    );
}

function EvidenceImage({ src, label }) {
    return (
        <>
            <div className="flex h-52 items-center justify-center rounded-lg bg-black">
                {src ? (
                    <img
                        src={`data:image/jpeg;base64,${src}`}
                        alt={label}
                        className="max-h-full max-w-full object-contain"
                    />
                ) : (
                    <span className="text-sm text-[var(--color-text-subtle)]">
                        Không có ảnh
                    </span>
                )}
            </div>
            <p className="mt-2 text-xs font-semibold text-[var(--color-text-muted)]">
                {label}
            </p>
        </>
    );
}

function formatRecognitionIdentity(vehicle) {
    const resultId = vehicle.recognition_id ?? vehicle.id;
    const vehicleTrackId = vehicle.vehicle_track_id;
    const plateTrackId = vehicle.plate_track_id;
    const parts = [`Kết quả #${resultId}`];
    if (vehicleTrackId !== undefined && vehicleTrackId !== null)
        parts.push(`Xe #${vehicleTrackId}`);
    if (plateTrackId !== undefined && plateTrackId !== null)
        parts.push(`Biển số #${plateTrackId}`);
    return parts.join(" · ");
}
