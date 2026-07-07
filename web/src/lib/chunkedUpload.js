// Chunked upload helper — splits a large file into sub-limit parts so it can
// pass a body-size-limited proxy (Cloudflare free caps request bodies at 100 MB).
// The server reassembles the parts and returns the completion JSON.

export const CHUNK_SIZE = 16 * 1024 * 1024; // 16 MB per request
export const SINGLE_UPLOAD_MAX = 80 * 1024 * 1024; // ≤ this goes in a single POST

export function randomUploadId() {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Upload `file` in chunks and return the parsed JSON from the complete endpoint.
 *
 * @param {File}     file
 * @param {(path: string, options: object) => Promise<Response>} poster
 * @param {{chunk: string, complete: string, abort: string}} paths
 * @param {Record<string,string>} [fields]  extra form fields for /complete
 * @param {(fraction: number) => void} [onProgress]
 */
export async function uploadInChunks({
    file,
    poster,
    paths,
    fields = {},
    onProgress,
}) {
    const uploadId = randomUploadId();
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

    try {
        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const blob = file.slice(
                start,
                Math.min(start + CHUNK_SIZE, file.size),
            );
            const fd = new FormData();
            fd.append("upload_id", uploadId);
            fd.append("chunk_index", String(i));
            fd.append("total_chunks", String(totalChunks));
            fd.append("filename", file.name);
            fd.append("chunk", blob, file.name);

            const res = await poster(paths.chunk, { method: "POST", body: fd });
            if (!res.ok) {
                throw new Error(
                    `Tải mảnh ${i + 1}/${totalChunks} thất bại: ${res.statusText}`,
                );
            }
            if (onProgress) onProgress((i + 1) / totalChunks);
        }
    } catch (err) {
        // Best-effort cleanup of partial chunks on the server.
        try {
            await poster(`${paths.abort}/${uploadId}`, { method: "DELETE" });
        } catch {
            /* ignore */
        }
        throw err;
    }

    const fd = new FormData();
    fd.append("upload_id", uploadId);
    fd.append("total_chunks", String(totalChunks));
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);

    const res = await poster(paths.complete, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Ghép video thất bại: ${res.statusText}`);
    return res.json();
}
