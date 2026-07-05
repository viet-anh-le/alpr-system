import { useEffect, useRef } from 'react'
import { API_BASE } from '../apiClient'

export function useStream(jobId, {
  onVehicle,
  onRejectedVehicle,
  onProgress,
  onComplete,
  onError,
  onFrame,
  onPreprocessedFrame,
}) {
  useEffect(() => {
    if (!jobId) return

    const abortController = new AbortController()

    async function startStream() {
      try {
        const res = await fetch(`${API_BASE}/stream/${jobId}`, {
          headers: { 'ngrok-skip-browser-warning': '1' },
          credentials: 'include',
          signal: abortController.signal
        })
        
        if (!res.ok) {
          onError('Lỗi kết nối với server')
          return
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // Giữ lại phần chưa hoàn chỉnh

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim()
              if (!dataStr) continue
              try {
                const ev = JSON.parse(dataStr)
                switch (ev.type) {
                  case 'progress': onProgress(ev);       break
                  case 'vehicle':  onVehicle(ev);        break
                  case 'rejected_vehicle': onRejectedVehicle?.(ev); break
                  case 'frame':    onFrame?.(ev);        break
                  case 'preprocessed_frame': onPreprocessedFrame?.(ev); break
                  case 'complete': onComplete(ev); abortController.abort(); break
                  case 'error':    onError(ev.message);  abortController.abort(); break
                  case 'ping':     break
                }
              } catch (e) {
                console.error('Lỗi parse SSE:', e)
              }
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          onError('Mất kết nối với server')
        }
      }
    }

    startStream()

    return () => abortController.abort()
  }, [jobId]) // eslint-disable-line react-hooks/exhaustive-deps
}
