import { useEffect, useRef } from 'react'

export function useStream(jobId, { onVehicle, onRejectedVehicle, onProgress, onComplete, onError, onFrame }) {
  const esRef = useRef(null)

  useEffect(() => {
    if (!jobId) return

    const es = new EventSource(`/stream/${jobId}`)
    esRef.current = es

    es.onmessage = (e) => {
      const ev = JSON.parse(e.data)
      switch (ev.type) {
        case 'progress': onProgress(ev);       break
        case 'vehicle':  onVehicle(ev);        break
        case 'rejected_vehicle': onRejectedVehicle?.(ev); break
        case 'frame':    onFrame?.(ev);        break
        case 'complete': onComplete(ev); es.close(); break
        case 'error':    onError(ev.message);  es.close(); break
        case 'ping':     break
      }
    }

    es.onerror = () => {
      onError('Mất kết nối với server')
      es.close()
    }

    return () => es.close()
  }, [jobId]) // eslint-disable-line react-hooks/exhaustive-deps
}
