import { apiFetch } from "../apiClient";
import { SINGLE_UPLOAD_MAX, uploadInChunks } from "../lib/chunkedUpload";

function toResult(data) {
    return {
        jobId: data.job_id,
        preprocessMode: data.preprocess_mode,
        ocrBackend: data.ocr_backend,
        processedVideoExpected: Boolean(data.processed_video_expected),
    };
}

export function useUpload() {
    // onProgress(fraction 0..1) is optional; reports upload progress.
    async function uploadVideo(
        file,
        preprocessMode = "none",
        ocrBackend = "default",
        onProgress,
    ) {
        if (file.size > SINGLE_UPLOAD_MAX) {
            const data = await uploadInChunks({
                file,
                poster: apiFetch,
                paths: {
                    chunk: "/upload/chunk",
                    complete: "/upload/complete",
                    abort: "/upload/chunk",
                },
                fields: {
                    preprocess_mode: preprocessMode,
                    ocr_backend: ocrBackend,
                },
                onProgress,
            });
            return toResult(data);
        }

        const fd = new FormData();
        fd.append("file", file);
        fd.append("preprocess_mode", preprocessMode);
        fd.append("ocr_backend", ocrBackend);
        const res = await apiFetch("/upload", { method: "POST", body: fd });
        if (!res.ok) throw new Error(`Upload thất bại: ${res.statusText}`);
        if (onProgress) onProgress(1);
        return toResult(await res.json());
    }

    return { uploadVideo };
}
