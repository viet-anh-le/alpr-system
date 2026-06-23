import { useCallback, useEffect, useRef, useState } from 'react'

import { Badge, Button, Toast } from '../ui'
import SourceSelector from './SourceSelector'
import LiveViewer from './LiveViewer'
import UploadViewer from './UploadViewer'
import IncidentsPanel from './IncidentsPanel'
import useIncidentStream from '../../hooks/monitor/useIncidentStream'
import { postMark } from '../../hooks/monitor/useMark'

function cleanupMonitorSession(session) {
  if (!session?.sessionId) return
  const url = session.mode === 'live'
    ? `/monitor/live/${session.sessionId}`
    : session.mode === 'upload'
      ? `/monitor/upload/${session.sessionId}`
      : null
  if (!url) return
  fetch(url, { method: 'DELETE', keepalive: true }).catch(() => {})
}

export default function MonitorPage() {
  const [session, setSession] = useState(null)
  const [incidents, setIncidents] = useState({})
  const [error, setError] = useState(null)
  const [pendingAction, setPendingAction] = useState(null)
  const cleanupTimersRef = useRef(new Map())

  const handleEvent = useCallback((event) => {
    const id = event.incident_id
    if (!id) return

    setIncidents((prev) => {
      const current = prev[id] || {
        id,
        status: 'pending',
        vehicles: {},
        markedAt: new Date().toISOString(),
        windowStartSec: 0,
        windowEndSec: 0,
      }

      switch (event.type) {
        case 'incident_started':
          return {
            ...prev,
            [id]: {
              ...current,
              status: 'processing',
              sourceType: event.source_type,
              windowStartSec: event.window_start_sec,
              windowEndSec: event.window_end_sec,
              framesCount: event.frames_count,
            },
          }
        case 'incident_progress':
          return { ...prev, [id]: { ...current, pct: event.pct } }
        case 'incident_vehicle':
          return {
            ...prev,
            [id]: { ...current, vehicles: { ...current.vehicles, [event.id]: event } },
          }
        case 'incident_rejected_vehicle':
          return {
            ...prev,
            [id]: { ...current, rejected: { ...(current.rejected || {}), [event.id]: event } },
          }
        case 'incident_complete':
          return {
            ...prev,
            [id]: {
              ...current,
              status: 'completed',
              durationMs: event.duration_ms,
              totalVehicles: event.total_vehicles,
            },
          }
        case 'incident_error':
          return { ...prev, [id]: { ...current, status: 'failed', error: event.message } }
        default:
          return prev
      }
    })
  }, [])

  useIncidentStream(session?.sessionId, handleEvent)

  useEffect(() => {
    if (!session?.sessionId) return undefined
    const key = `${session.mode}:${session.sessionId}`
    const pendingCleanup = cleanupTimersRef.current.get(key)
    if (pendingCleanup) {
      window.clearTimeout(pendingCleanup)
      cleanupTimersRef.current.delete(key)
    }

    const handlePageHide = () => cleanupMonitorSession(session)
    window.addEventListener('pagehide', handlePageHide)
    return () => {
      window.removeEventListener('pagehide', handlePageHide)
      const timer = window.setTimeout(() => {
        cleanupTimersRef.current.delete(key)
        cleanupMonitorSession(session)
      }, 250)
      cleanupTimersRef.current.set(key, timer)
    }
  }, [session?.sessionId, session?.mode])

  const handleConnectLive = async (rtspUrl, ocrBackend = 'default') => {
    setError(null)
    setPendingAction('live')
    try {
      const response = await fetch('/monitor/live/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: rtspUrl, ocr_backend: ocrBackend }),
      })
      if (!response.ok) {
        setError(`Không kết nối được camera: ${await response.text()}`)
        return
      }
      const data = await response.json()
      setSession({ mode: 'live', sessionId: data.session_id, whepUrl: data.whep_url, mjpegUrl: data.mjpeg_url, rtspUrl, ocrBackend })
      setIncidents({})
    } catch (err) {
      setError(`Không kết nối được camera: ${err.message}`)
    } finally {
      setPendingAction(null)
    }
  }

  const handleSelectFile = async (file, preprocessMode = 'none', ocrBackend = 'default') => {
    setError(null)
    setPendingAction('upload')
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('preprocess_mode', preprocessMode)
      formData.append('ocr_backend', ocrBackend)
      const response = await fetch('/monitor/upload', { method: 'POST', body: formData })
      if (!response.ok) {
        setError(`Upload monitor thất bại: ${await response.text()}`)
        return
      }
      const data = await response.json()
      setSession({
        mode: 'upload',
        sessionId: data.session_id,
        videoUrl: data.video_url,
        preprocessMode: data.preprocess_mode,
        ocrBackend: data.ocr_backend,
        file,
      })
      setIncidents({})
    } catch (err) {
      setError(`Không mở được video: ${err.message}`)
    } finally {
      setPendingAction(null)
    }
  }

  const handleMarkLive = async () => {
    try {
      const id = await postMark(session.sessionId, { mode: 'live' })
      setIncidents((prev) => (
        prev[id]
          ? prev
          : {
              ...prev,
              [id]: {
                id,
                status: 'pending',
                markedAt: new Date().toISOString(),
                windowStartSec: 0,
                windowEndSec: 0,
                vehicles: {},
              },
            }
      ))
    } catch (err) {
      setError(`Mark failed: ${err.message}`)
    }
  }

  const handleMarkUpload = async (start, end) => {
    try {
      const id = await postMark(session.sessionId, { mode: 'upload', t_start: start, t_end: end })
      setIncidents((prev) => (
        prev[id]
          ? prev
          : {
              ...prev,
              [id]: {
                id,
                status: 'pending',
                markedAt: new Date().toISOString(),
                windowStartSec: start,
                windowEndSec: end,
                vehicles: {},
              },
            }
      ))
    } catch (err) {
      setError(`Mark failed: ${err.message}`)
    }
  }

  const disconnect = () => {
    cleanupMonitorSession(session)
    setSession(null)
    setIncidents({})
  }

  if (!session) {
    return (
      <>
        <SourceSelector
          onConnectLive={handleConnectLive}
          onSelectFile={handleSelectFile}
          isConnectingLive={pendingAction === 'live'}
          isOpeningVideo={pendingAction === 'upload'}
        />
        <Toast message={error} onDismiss={() => setError(null)} />
      </>
    )
  }

  return (
    <>
      <div className="evidence-grid">
        <div className="space-y-4">
          <section className="surface-panel p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="section-label">Active monitor session</p>
                <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                  {session.mode === 'live' ? session.rtspUrl : session.file?.name || 'Uploaded video'}
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge tone={session.mode === 'live' ? 'success' : 'info'}>{session.mode}</Badge>
                {session.mode === 'live' && (
                  <Button variant="danger" onClick={handleMarkLive}>Mark last 10s</Button>
                )}
                <Button variant="ghost" onClick={disconnect}>Đóng session</Button>
              </div>
            </div>
          </section>

          {session.mode === 'live' ? (
            <LiveViewer whepUrl={session.whepUrl} mjpegUrl={session.mjpegUrl} />
          ) : (
            <UploadViewer videoUrl={session.videoUrl} onMark={handleMarkUpload} />
          )}
        </div>
        <IncidentsPanel incidents={incidents} />
      </div>
      <Toast message={error} onDismiss={() => setError(null)} />
    </>
  )
}
