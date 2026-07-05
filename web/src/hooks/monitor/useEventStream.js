import { useEffect } from 'react'
import { API_BASE } from '../../apiClient'

/**
 * Subscribes to /monitor/{sessionId}/events/stream and calls onEvent
 * for each event received. Closes the connection on unmount.
 */
export default function useEventStream(sessionId, onEvent) {
  useEffect(() => {
    if (!sessionId) return undefined

    const abortController = new AbortController()

    async function startStream() {
      try {
        const res = await fetch(`${API_BASE}/monitor/${sessionId}/events/stream`, {
          headers: { 'ngrok-skip-browser-warning': '1' },
          credentials: 'include',
          signal: abortController.signal
        })

        if (!res.ok) return

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim()
              if (!dataStr) continue
              try {
                const ev = JSON.parse(dataStr)
                if (ev.type === 'ping') continue
                onEvent(ev)
              } catch (e) {
                console.error('Lỗi parse SSE:', e)
              }
            }
          }
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Stream error:', err)
        }
      }
    }

    startStream()

    return () => abortController.abort()
  }, [sessionId, onEvent])
}
