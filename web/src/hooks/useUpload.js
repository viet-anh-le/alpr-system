import { apiFetch } from '../apiClient'

export function useUpload() {
  async function uploadVideo(file, preprocessMode = 'none', ocrBackend = 'default') {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('preprocess_mode', preprocessMode)
    fd.append('ocr_backend', ocrBackend)

    const res = await apiFetch('/upload', { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`Upload thất bại: ${res.statusText}`)

    const { job_id } = await res.json()
    return job_id
  }

  return { uploadVideo }
}
