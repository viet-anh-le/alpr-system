export const HISTORY_VEHICLE_FILTER_OPTIONS = [
    { value: "all", label: "Tất cả phương tiện" },
    { value: "car", label: "Ô tô" },
    { value: "motorbike", label: "Motorbike" },
    { value: "bus", label: "Xe buýt" },
    { value: "truck", label: "Xe tải" },
];

export function displayPlateText(text) {
    return (text || "").replaceAll("[SEP]", " ").trim();
}

export function buildSessionsPath({ page = 1, limit = 20 } = {}) {
    const safeLimit = positiveInteger(limit, 20);
    const safePage = positiveInteger(page, 1);
    const params = new URLSearchParams({
        limit: String(safeLimit),
        offset: String((safePage - 1) * safeLimit),
    });
    return `/sessions?${params.toString()}`;
}

export function buildRecordsPath({
    page = 1,
    limit = 12,
    sessionId = "",
    plate = "",
    vehicleClass = "",
} = {}) {
    const safeLimit = positiveInteger(limit, 12);
    const safePage = positiveInteger(page, 1);
    const params = new URLSearchParams({
        limit: String(safeLimit),
        offset: String((safePage - 1) * safeLimit),
    });
    if (sessionId) params.set("session_id", sessionId);
    if (plate.trim()) params.set("plate", plate.trim());
    if (vehicleClass && vehicleClass !== "all")
        params.set("vehicle_class", vehicleClass);
    return `/records?${params.toString()}`;
}

export function pageInfo(total = 0, page = 1, limit = 12) {
    const safeTotal = Math.max(0, Number(total) || 0);
    const safeLimit = positiveInteger(limit, 12);
    const totalPages = Math.max(1, Math.ceil(safeTotal / safeLimit));
    const currentPage = Math.min(
        Math.max(positiveInteger(page, 1), 1),
        totalPages,
    );
    const start = safeTotal === 0 ? 0 : (currentPage - 1) * safeLimit + 1;
    const end = Math.min(currentPage * safeLimit, safeTotal);
    return {
        total: safeTotal,
        page: currentPage,
        totalPages,
        start,
        end,
        hasPrev: currentPage > 1,
        hasNext: currentPage < totalPages,
    };
}

export function normalizeHistorySummary(summary = {}) {
    return {
        totalRecords: Number(summary.total_records) || 0,
        uniquePlates: Number(summary.unique_plates) || 0,
        vehicleCounts: Array.isArray(summary.vehicle_counts)
            ? summary.vehicle_counts
            : [],
        topPlates: Array.isArray(summary.top_plates)
            ? summary.top_plates.map((item) => ({
                  plateText: displayPlateText(item.plate_text),
              }))
            : [],
    };
}

function positiveInteger(value, fallback) {
    const numberValue = Number(value);
    return Number.isInteger(numberValue) && numberValue > 0
        ? numberValue
        : fallback;
}
