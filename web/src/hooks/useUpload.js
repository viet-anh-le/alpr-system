import { apiFetch } from '../apiClient'

export function useUpload() {
  async function uploadVideo(file, preprocessMode = 'none', ocrBackend = 'default') {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('preprocess_mode', preprocessMode)
    fd.append('ocr_backend', ocrBackend)

    const res = await apiFetch('/upload', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`Upload thất bại: ${res.statusText}`)

    const {
      job_id,
      preprocess_mode,
      ocr_backend,
      processed_video_expected,
    } = await res.json()
    return {
      jobId: job_id,
      preprocessMode: preprocess_mode,
      ocrBackend: ocr_backend,
      processedVideoExpected: Boolean(processed_video_expected),
    }
  }

  return { uploadVideo }
}
