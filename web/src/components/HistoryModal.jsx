import { useEffect, useState } from "react";

import { apiJson } from "../apiClient";
import TrackBufferModal from "./TrackBufferModal";
import {
    HISTORY_VEHICLE_FILTER_OPTIONS,
    buildRecordsPath,
    buildSessionsPath,
    displayPlateText,
    normalizeHistorySummary,
    pageInfo,
} from "./historyControls";
import {
    Badge,
    Button,
    Drawer,
    EmptyState,
    Select,
    Skeleton,
    TextInput,
    cx,
} from "./ui";
import { VEHICLE_LABEL } from "./workbench/constants";

const SESSION_PAGE_SIZE = 20;
const RECORD_PAGE_SIZE = 12;

export default function HistoryModal({ open, onClose }) {
    const [jobs, setJobs] = useState([]);
    const [selectedJobId, setSelectedJobId] = useState("");
    const [vehicles, setVehicles] = useState([]);
    const [summary, setSummary] = useState(normalizeHistorySummary());
    const [sessionsTotal, setSessionsTotal] = useState(0);
    const [recordsTotal, setRecordsTotal] = useState(0);
    const [sessionPage, setSessionPage] = useState(1);
    const [recordPage, setRecordPage] = useState(1);
    const [draftPlate, setDraftPlate] = useState("");
    const [draftVehicleClass, setDraftVehicleClass] = useState("all");
    const [filters, setFilters] = useState({
        plate: "",
        vehicleClass: "all",
    });
    const [loadingJobs, setLoadingJobs] = useState(false);
    const [loadingVehicles, setLoadingVehicles] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!open) return;
        async function fetchJobs() {
            setLoadingJobs(true);
            setError(null);
            try {
                const data = await apiJson(
                    buildSessionsPath({
                        page: sessionPage,
                        limit: SESSION_PAGE_SIZE,
                    }),
                );
                const items = data.items || [];
                setJobs(items);
                setSessionsTotal(data.total ?? items.length);
                setSelectedJobId((current) => {
                    if (!items.length) return "";
                    const currentStillVisible = items.some(
                        (job) => job.session_id === current,
                    );
                    return currentStillVisible ? current : items[0].session_id;
                });
            } catch (err) {
                setError(err.message);
            } finally {
                setLoadingJobs(false);
            }
        }
        fetchJobs();
    }, [open, sessionPage]);

    useEffect(() => {
        if (!open) return;
        if (!selectedJobId) {
            setVehicles([]);
            setRecordsTotal(0);
            setSummary(normalizeHistorySummary());
            return;
        }
        async function fetchVehicles() {
            setVehicles([]);
            setLoadingVehicles(true);
            setError(null);
            try {
                const data = await apiJson(
                    buildRecordsPath({
                        page: recordPage,
                        limit: RECORD_PAGE_SIZE,
                        sessionId: selectedJobId,
                        plate: filters.plate,
                        vehicleClass: filters.vehicleClass,
                    }),
                );
                const items = data.items || [];
                setVehicles(items);
                setRecordsTotal(data.total ?? items.length);
                setSummary(normalizeHistorySummary(data.summary));
            } catch (err) {
                setError(err.message);
            } finally {
                setLoadingVehicles(false);
            }
        }
        fetchVehicles();
    }, [open, selectedJobId, filters, recordPage]);

    const sessionPageInfo = pageInfo(
        sessionsTotal,
        sessionPage,
        SESSION_PAGE_SIZE,
    );
    const recordPageInfo = pageInfo(recordsTotal, recordPage, RECORD_PAGE_SIZE);
    const selectedJob = jobs.find((job) => job.session_id === selectedJobId);
    const selectedScopeLabel = selectedJobId
        ? selectedJob?.source_filename || `Phiên #${selectedJobId}`
        : "Chọn phiên xử lý";

    const applyFilters = (event) => {
        event.preventDefault();
        setRecordPage(1);
        setFilters({
            plate: draftPlate.trim(),
            vehicleClass: draftVehicleClass,
        });
    };

    const clearFilters = () => {
        setDraftPlate("");
        setDraftVehicleClass("all");
        setRecordPage(1);
        setFilters({ plate: "", vehicleClass: "all" });
    };

    const selectJob = (sessionId) => {
        setSelectedJobId(sessionId);
        setRecordPage(1);
    };

    return (
        <Drawer
            open={open}
            onClose={onClose}
            title="Lịch sử nhận dạng"
            description="Tra cứu phiên, biển số, loại phương tiện và bộ đệm bằng chứng đã lưu."
            className="max-w-7xl"
        >
            <div className="grid min-h-full lg:grid-cols-[330px_1fr]">
                <aside className="border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] lg:border-b-0 lg:border-r">
                    <div className="panel-header">
                        <div>
                            <p className="section-label">Phiên xử lý</p>
                            <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                                {sessionsTotal} phiên đã lưu
                            </p>
                        </div>
                    </div>
                    <div className="space-y-3 p-3">
                        <div className="max-h-[42vh] overflow-y-auto lg:max-h-[calc(100vh-17rem)]">
                            {loadingJobs ? (
                                <div className="space-y-2">
                                    {Array.from({ length: 5 }).map(
                                        (_, index) => (
                                            <Skeleton
                                                key={index}
                                                className="h-20"
                                            />
                                        ),
                                    )}
                                </div>
                            ) : jobs.length === 0 ? (
                                <EmptyState title="Chưa có phiên">
                                    Sau khi xử lý video thành công, phiên và bản
                                    ghi nhận dạng sẽ xuất hiện ở đây.
                                </EmptyState>
                            ) : (
                                <div className="space-y-2">
                                    {jobs.map((job) => (
                                        <SessionButton
                                            key={job.session_id}
                                            job={job}
                                            active={
                                                selectedJobId === job.session_id
                                            }
                                            onClick={() =>
                                                selectJob(job.session_id)
                                            }
                                        />
                                    ))}
                                </div>
                            )}
                        </div>

                        <Pagination
                            info={sessionPageInfo}
                            itemLabel="phiên"
                            onPageChange={setSessionPage}
                        />
                    </div>
                </aside>

                <section className="min-h-0 bg-[var(--color-bg)]">
                    <div className="border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
                        <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
                            <div className="min-w-0">
                                <p className="section-label">Phiên đang xem</p>
                                <h3
                                    className="mt-1 truncate text-base font-bold text-[var(--color-text)]"
                                    title={selectedScopeLabel}
                                >
                                    {selectedScopeLabel}
                                </h3>
                            </div>
                            <form
                                className="grid gap-2 md:grid-cols-[minmax(180px,1fr)_220px_auto_auto]"
                                onSubmit={applyFilters}
                            >
                                <TextInput
                                    aria-label="Tìm theo biển số"
                                    placeholder="Tìm biển số..."
                                    value={draftPlate}
                                    onChange={(event) =>
                                        setDraftPlate(event.target.value)
                                    }
                                />
                                <Select
                                    aria-label="Lọc loại phương tiện"
                                    value={draftVehicleClass}
                                    onChange={(event) =>
                                        setDraftVehicleClass(event.target.value)
                                    }
                                    options={HISTORY_VEHICLE_FILTER_OPTIONS}
                                />
                                <Button type="submit" variant="primary">
                                    Tìm
                                </Button>
                                <Button
                                    type="button"
                                    variant="secondary"
                                    onClick={clearFilters}
                                >
                                    Xóa lọc
                                </Button>
                            </form>
                        </div>
                    </div>

                    {error ? (
                        <div className="p-4">
                            <EmptyState title="Không tải được lịch sử">
                                {error}
                            </EmptyState>
                        </div>
                    ) : (
                        <div className="space-y-4 p-4">
                            <HistorySummary summary={summary} />

                            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                <div>
                                    <p className="section-label">
                                        Bản ghi nhận dạng
                                    </p>
                                </div>
                                <Pagination
                                    info={recordPageInfo}
                                    itemLabel="bản ghi"
                                    onPageChange={setRecordPage}
                                />
                            </div>

                            {loadingVehicles ? (
                                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                                    {Array.from({ length: 6 }).map(
                                        (_, index) => (
                                            <Skeleton
                                                key={index}
                                                className="h-64"
                                            />
                                        ),
                                    )}
                                </div>
                            ) : vehicles.length === 0 ? (
                                <EmptyState title="Không có bản ghi phù hợp">
                                    Thử bỏ bớt điều kiện query hoặc chọn một
                                    phiên khác để kiểm tra.
                                </EmptyState>
                            ) : (
                                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                                    {vehicles.map((vehicle) => (
                                        <HistoryRecord
                                            key={`${vehicle.session_id}-${vehicle.track_id}`}
                                            vehicle={vehicle}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </section>
            </div>
        </Drawer>
    );
}

function SessionButton({ job, active, onClick }) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={cx(
                "w-full rounded-[var(--radius-control)] border p-3 text-left transition-colors duration-200",
                active
                    ? "border-cyan-300/45 bg-cyan-300/10"
                    : "border-[var(--color-border)] bg-black/10 hover:bg-white/5",
            )}
        >
            <div className="flex items-center justify-between gap-2">
                <p
                    className="min-w-0 truncate text-sm font-semibold"
                    title={job.source_filename}
                >
                    {job.source_filename}
                </p>
                <Badge
                    tone={
                        job.status === "completed"
                            ? "success"
                            : job.status === "failed"
                              ? "danger"
                              : "info"
                    }
                >
                    {getJobStatusLabel(job.status)}
                </Badge>
            </div>
            <p
                className="mt-2 data-font truncate text-[11px] text-[var(--color-text-subtle)]"
                title={job.session_id}
            >
                #{job.session_id}
            </p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                {new Date(job.created_at).toLocaleString("vi")} ·{" "}
                {job.total_records || 0} bản ghi
            </p>
        </button>
    );
}

function HistorySummary({ summary }) {
    return (
        <section className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
            <div className="grid gap-3 md:grid-cols-[repeat(3,minmax(0,1fr))]">
                <SummaryMetric label="Bản ghi" value={summary.totalRecords} />
                <SummaryMetric
                    label="Biển duy nhất"
                    value={summary.uniquePlates}
                    tone="success"
                />
                <SummaryMetric
                    label="Loại phương tiện"
                    value={summary.vehicleCounts.length}
                    tone="info"
                />
            </div>
        </section>
    );
}

function SummaryMetric({ label, value, tone = "neutral" }) {
    const toneClass = {
        neutral: "text-[var(--color-text)]",
        success: "text-emerald-100",
        info: "text-cyan-100",
    }[tone];
    return (
        <div className="rounded-[var(--radius-control)] border border-[var(--color-border)] bg-black/15 p-3">
            <p className="text-xs font-semibold text-[var(--color-text-subtle)]">
                {label}
            </p>
            <p className={cx("data-font mt-1 text-2xl font-bold", toneClass)}>
                {value}
            </p>
        </div>
    );
}

function Pagination({ info, itemLabel, onPageChange }) {
    return (
        <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--color-text-muted)]">
            <span className="data-font">
                {info.start}-{info.end}/{info.total} {itemLabel}
            </span>
            <span className="text-[var(--color-text-subtle)]">
                Trang {info.page}/{info.totalPages}
            </span>
            <Button
                size="sm"
                variant="ghost"
                disabled={!info.hasPrev}
                onClick={() => onPageChange(info.page - 1)}
            >
                Trước
            </Button>
            <Button
                size="sm"
                variant="ghost"
                disabled={!info.hasNext}
                onClick={() => onPageChange(info.page + 1)}
            >
                Sau
            </Button>
        </div>
    );
}

function HistoryRecord({ vehicle }) {
    const [bufferVehicle, setBufferVehicle] = useState(null);
    const confidence = Math.round((vehicle.plate_text_confidence || 0) * 100);
    const identity = formatRecognitionIdentity(vehicle);
    const clusters = Array.isArray(vehicle.clusters) ? vehicle.clusters : [];
    const plateText = displayPlateText(vehicle.plate_text);

    return (
        <>
            <article className="overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
                <div className="grid grid-cols-2 gap-px bg-[var(--color-border)]">
                    <HistoryImage
                        src={vehicle.vehicle_thumbnail_url}
                        alt="Ảnh phương tiện đối chiếu"
                    />
                    <HistoryImage
                        src={vehicle.best_plate_frame?.image_url}
                        alt="Ảnh biển số đối chiếu"
                        dark
                    />
                </div>
                <div className="space-y-3 p-3">
                    <div className="flex items-start justify-between gap-3">
                        <p
                            className="plate-font min-w-0 truncate text-lg font-bold tracking-widest"
                            title={plateText}
                        >
                            {plateText || "—"}
                        </p>
                        <Badge
                            tone={
                                confidence >= 90
                                    ? "success"
                                    : confidence >= 70
                                      ? "warning"
                                      : "danger"
                            }
                        >
                            {confidence}%
                        </Badge>
                    </div>
                    <p className="text-xs text-[var(--color-text-muted)]">
                        {identity} ·{" "}
                        {VEHICLE_LABEL[vehicle.vehicle_class] ||
                            vehicle.vehicle_class ||
                            "Phương tiện"}{" "}
                        · {formatOcrMethod(vehicle.ocr_method)}
                    </p>
                    <p className="data-font text-[11px] text-[var(--color-text-subtle)]">
                        Khung {vehicle.first_seen_frame ?? "—"} →{" "}
                        {vehicle.last_seen_frame ?? "—"}
                    </p>
                    <Button
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                            setBufferVehicle(toModalVehicle(vehicle))
                        }
                    >
                        Bộ đệm theo vết
                    </Button>

                    {clusters.length > 1 && (
                        <div className="space-y-2 border-t border-[var(--color-border)] pt-3">
                            <p className="section-label">Cụm OCR đã lưu</p>
                            {clusters.map((cluster) => (
                                <HistoryCluster
                                    key={cluster.cluster_index}
                                    vehicle={vehicle}
                                    cluster={cluster}
                                    onInspect={setBufferVehicle}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </article>

            {bufferVehicle && (
                <TrackBufferModal
                    vehicle={bufferVehicle}
                    jobId={vehicle.session_id}
                    onClose={() => setBufferVehicle(null)}
                />
            )}
        </>
    );
}

function HistoryCluster({ vehicle, cluster, onInspect }) {
    const clusterConfidence = Math.round(
        (cluster.plate_text_confidence || 0) * 100,
    );
    const plateText = displayPlateText(cluster.plate_text);

    return (
        <div className="rounded-lg border border-[var(--color-border)] bg-black/15 p-2">
            <div className="flex items-center justify-between gap-2">
                <Badge tone="neutral">
                    Cụm {(cluster.cluster_index ?? 0) + 1}
                </Badge>
                <span className="text-[10px] text-[var(--color-text-subtle)]">
                    {cluster.frame_count || cluster.track_buffer?.length || 0}{" "}
                    khung · {clusterConfidence}%
                </span>
            </div>
            <div className="mt-2 flex items-center gap-3">
                <div className="h-14 w-24 flex-shrink-0 overflow-hidden rounded bg-black">
                    {cluster.best_plate_frame?.image_url ? (
                        <img
                            src={cluster.best_plate_frame.image_url}
                            alt={`Cụm ${(cluster.cluster_index ?? 0) + 1}`}
                            className="h-full w-full object-contain"
                        />
                    ) : (
                        <span className="flex h-full items-center justify-center text-[10px] text-[var(--color-text-subtle)]">
                            Không có ảnh
                        </span>
                    )}
                </div>
                <div className="min-w-0 flex-1">
                    <p
                        className="plate-font truncate text-sm font-bold tracking-widest"
                        title={plateText}
                    >
                        {plateText || "—"}
                    </p>
                    <Button
                        className="mt-2"
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                            onInspect(toModalVehicle(vehicle, cluster))
                        }
                    >
                        Xem bộ đệm
                    </Button>
                </div>
            </div>
        </div>
    );
}

function toModalVehicle(record, cluster = null) {
    if (!cluster) {
        return {
            ...record,
            id: record.track_id,
            recognition_id: record.track_id,
            cls: record.vehicle_class,
            plate: record.plate_text,
            chars: record.chars || [],
            vote_summary: record.ocr_vote_summary,
            vehicle_b64: record.vehicle_thumbnail_url,
            plate_b64: record.best_plate_frame?.image_url,
        };
    }

    return {
        ...record,
        ...cluster,
        id: record.track_id,
        recognition_id: record.track_id,
        track_id: record.track_id,
        vehicle_track_id: record.vehicle_track_id,
        plate_track_id: record.plate_track_id,
        cls: record.vehicle_class,
        plate: cluster.plate_text,
        chars: cluster.chars || [],
        vote_summary: cluster.ocr_vote_summary,
        vehicle_b64: record.vehicle_thumbnail_url,
        plate_b64: cluster.best_plate_frame?.image_url,
    };
}

function formatRecognitionIdentity(vehicle) {
    const parts = [`Kết quả #${vehicle.track_id}`];
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

function getJobStatusLabel(status) {
    if (status === "completed") return "Hoàn tất";
    if (status === "failed") return "Có lỗi";
    if (status === "processing") return "Đang xử lý";
    return "Đang chờ";
}

function formatOcrMethod(value) {
    if (!value) return "OCR";
    const labels = {
        realtime_buffer: "Bộ đệm thời gian thực",
        default: "SmallLPR-Line-CTC (mặc định)",
        smalllpr_line_ctc: "SmallLPR-Line-CTC",
        vietnamese_yolov5: "YOLOv5 Việt Nam",
    };
    return labels[value] || value.replaceAll("_", " ");
}

function HistoryImage({ src, alt, dark = false }) {
    return (
        <div
            className={cx(
                "flex h-32 items-center justify-center",
                dark ? "bg-black" : "bg-black/30",
            )}
        >
            {src ? (
                <img
                    src={src}
                    alt={alt}
                    className="max-h-full max-w-full object-contain"
                />
            ) : (
                <span className="text-xs text-[var(--color-text-subtle)]">
                    Không có ảnh
                </span>
            )}
        </div>
    );
}
