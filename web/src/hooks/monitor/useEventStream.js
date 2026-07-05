import { useEffect } from 'react'
import { API_BASE } from '../../apiClient'

/**
 * Subscribes to /monitor/{sessionId}/events/stream and calls onEvent
 * for each event received. Closes the EventSource on unmount.
 */
export default function useEventStream(sessionId, onEvent) {
  useEffect(() => {
    if (!sessionId) return undefined
    const es = new EventSource(`${API_BASE}/monitor/${sessionId}/events/stream`, { withCredentials: true })
    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data)
        if (ev.type === 'ping') return
        onEvent(ev)
      } catch (e) {
        console.error('SSE parse error', e)
      }
    }
    es.onerror = () => { /* EventSource auto-reconnects */ }
    return () => es.close()
  }, [sessionId, onEvent])
}
