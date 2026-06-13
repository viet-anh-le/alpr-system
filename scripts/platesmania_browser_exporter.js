/*
 * Browser-side Platesmania Vietnam gallery exporter.
 *
 * Usage:
 * 1. Open https://platesmania.com/vn/gallery in your browser and pass the site verification manually.
 * 2. Open DevTools Console.
 * 3. Paste this whole file, adjust MODE/ranges/DELAY_MS if needed, and press Enter.
 * 4. Save the downloaded gallery_records.jsonl into data/raw/platesmania_vn/html_pages/.
 *
 * The exporter stores only metadata needed by collect_platesmania_vn_dataset.py:
 * full-frame vehicle image URL, detail URL, generated plate URL for attribution/debug,
 * and plate text from the generated plate image alt. It does not download images.
 */
(async () => {
    const MODE = "province-search"; // "province-search" or "gallery"
    const GALLERY_START_INDEX = 0;
    const GALLERY_END_INDEX = 100;
    const PROVINCE_START = 51;
    const PROVINCE_END = 99;
    const SEARCH_START_MIN = 0;
    const SEARCH_START_MAX = 100;
    const DELAY_MS = 2000;
    const STOP_AFTER_EMPTY_PAGES = 2;
    const STOP_AFTER_DUPLICATE_PAGES = 2;

    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    const galleryUrl = (index) => {
        if (index === 0) {
            return "https://platesmania.com/vn/gallery";
        }
        return `https://platesmania.com/vn/gallery-${index}`;
    };

    const provinceSearchUrl = (nomer, start) =>
        `https://platesmania.com/vn/gallery.php?&nomer=${nomer}&start=${start}`;

    const pageRequests = function* () {
        if (MODE === "province-search") {
            for (
                let nomer = PROVINCE_START;
                nomer <= PROVINCE_END;
                nomer += 1
            ) {
                for (
                    let start = SEARCH_START_MIN;
                    start <= SEARCH_START_MAX;
                    start += 1
                ) {
                    yield {
                        label: `nomer=${nomer} start=${start}`,
                        url: provinceSearchUrl(nomer, start),
                    };
                }
            }
            return;
        }

        for (
            let index = GALLERY_START_INDEX;
            index <= GALLERY_END_INDEX;
            index += 1
        ) {
            yield { label: `gallery-${index}`, url: galleryUrl(index) };
        }
    };

    const normalizePlate = (value) =>
        value.trim().toUpperCase().replace(/\s+/g, " ");

    const safeRecordId = (detailUrl) => {
        const match = detailUrl.match(/\/vn\/(nomer\d+)/);
        return match
            ? match[1]
            : `plate_${Math.random().toString(36).slice(2, 14)}`;
    };

    const isPlateText = (value) => {
        const normalized = normalizePlate(value);
        return /\d/.test(normalized) && /^[0-9A-ZĐ. -]+$/.test(normalized);
    };

    const parsePage = (html, pageUrl) => {
        const doc = new DOMParser().parseFromString(html, "text/html");
        const grouped = new Map();

        for (const anchor of doc.querySelectorAll('a[href*="/vn/nomer"]')) {
            const href = new URL(anchor.getAttribute("href"), pageUrl).href;
            const img = anchor.querySelector("img");
            if (!img) continue;

            const srcValue =
                img.getAttribute("src") ||
                img.getAttribute("data-src") ||
                img.getAttribute("data-original") ||
                img.getAttribute("data-lazy-src");
            if (!srcValue) continue;

            const src = new URL(srcValue, pageUrl).href;
            const alt = img.getAttribute("alt") || "";
            if (!grouped.has(href)) grouped.set(href, []);
            grouped.get(href).push({ src, alt });
        }

        const records = [];
        for (const [detailUrl, images] of grouped.entries()) {
            const vehicle = images.find(
                (item) =>
                    /\.(jpe?g|png|webp|bmp)$/i.test(
                        new URL(item.src).pathname,
                    ) && !new URL(item.src).pathname.includes("/inf/"),
            );
            const plateRef = images.find(
                (item) =>
                    new URL(item.src).pathname.includes("/inf/") &&
                    isPlateText(item.alt),
            );
            if (!vehicle || !plateRef) continue;

            records.push({
                record_id: safeRecordId(detailUrl),
                page_url: pageUrl,
                detail_url: detailUrl,
                vehicle_image_url: vehicle.src,
                plate_ref_url: plateRef.src,
                plate_text_raw: plateRef.alt.trim(),
                plate_text_normalized: normalizePlate(plateRef.alt),
            });
        }
        return records;
    };

    const allRecords = [];
    const seenRecordIds = new Set();
    const seenVehicleImageUrls = new Set();
    let emptyPages = 0;
    let duplicatePages = 0;
    for (const request of pageRequests()) {
        const requestedUrl = request.url;
        const response = await fetch(requestedUrl, {
            credentials: "include",
            headers: { Accept: "text/html,application/xhtml+xml" },
        });
        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status} while fetching ${requestedUrl}`,
            );
        }

        const html = await response.text();
        if (
            /killbot|user verification|verify you are human|checking your browser/i.test(
                html,
            )
        ) {
            throw new Error(`Verification page returned for ${requestedUrl}`);
        }

        const pageUrl = response.url || requestedUrl;
        if (pageUrl !== requestedUrl) {
            console.warn(`Redirected: ${requestedUrl} -> ${pageUrl}`);
        }

        const records = parsePage(html, pageUrl);
        const newRecords = records.filter(
            (record) =>
                !seenRecordIds.has(record.record_id) &&
                !seenVehicleImageUrls.has(record.vehicle_image_url),
        );
        console.log(
            `${request.label}: ${records.length} records, ${newRecords.length} new`,
        );
        for (const record of newRecords) {
            seenRecordIds.add(record.record_id);
            seenVehicleImageUrls.add(record.vehicle_image_url);
            allRecords.push(record);
        }

        if (records.length === 0) {
            emptyPages += 1;
            if (MODE === "gallery" && emptyPages >= STOP_AFTER_EMPTY_PAGES)
                break;
        } else {
            emptyPages = 0;
        }

        if (records.length > 0 && newRecords.length === 0) {
            duplicatePages += 1;
            console.warn(`Duplicate page records at ${requestedUrl}`);
            if (
                MODE === "gallery" &&
                duplicatePages >= STOP_AFTER_DUPLICATE_PAGES
            )
                break;
        } else {
            duplicatePages = 0;
        }

        if (DELAY_MS > 0) {
            await sleep(DELAY_MS);
        }
    }

    const jsonl =
        allRecords.map((record) => JSON.stringify(record)).join("\n") + "\n";
    const blob = new Blob([jsonl], {
        type: "application/x-ndjson;charset=utf-8",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "gallery_records.jsonl";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
    console.log(
        `Exported ${allRecords.length} records to gallery_records.jsonl`,
    );
})();
