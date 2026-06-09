import { useCallback, useEffect, useRef, useState } from 'react'

import SourceSelector  from './SourceSelector'
import LiveViewer      from './LiveViewer'
import UploadViewer    from './UploadViewer'
import IncidentsPanel  from './IncidentsPanel'
import useIncidentStream from '../../hooks/monitor/useIncidentStream'
import { postMark }    from '../../hooks/monitor/useMark'

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
  const [session,   setSession]   = useState(null)
  const [incidents, setIncidents] = useState({})
  const cleanupTimersRef = useRef(new Map())

  // SSE handler
  const handleEvent = useCallback((ev) => {
    const id = ev.incident_id
    if (!id) return

    setIncidents((prev) => {
      const cur = prev[id] || {
        id, status: 'pending', vehicles: {}, markedAt: new Date().toISOString(),
        windowStartSec: 0, windowEndSec: 0,
      }
      switch (ev.type) {
        case 'incident_started':
          return { ...prev, [id]: {
            ...cur,
            status: 'processing',
            sourceType: ev.source_type,
            windowStartSec: ev.window_start_sec,
            windowEndSec:   ev.window_end_sec,
            framesCount:    ev.frames_count,
          }}
        case 'incident_progress':
          return { ...prev, [id]: { ...cur, pct: ev.pct } }
        case 'incident_vehicle':
          return { ...prev, [id]: { ...cur,
            vehicles: { ...cur.vehicles, [ev.id]: ev }
          }}
        case 'incident_rejected_vehicle':
          return { ...prev, [id]: { ...cur,
            rejected: { ...(cur.rejected || {}), [ev.id]: ev }
          }}
        case 'incident_complete':
          return { ...prev, [id]: { ...cur, status: 'completed',
            durationMs: ev.duration_ms, totalVehicles: ev.total_vehicles }}
        case 'incident_error':
          return { ...prev, [id]: { ...cur, status: 'failed', error: ev.message } }
        default:
          return prev
      }
    })
  }, [])

  useIncidentStream(session?.sessionId, handleEvent)

  // Tear down monitor sessions on unmount, pagehide, or new-session.
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

  // ── Actions ────────────────────────────────────────────────────────────
  const handleConnectLive = async (rtspUrl, ocrBackend = 'default') => {
    const resp = await fetch('/monitor/live/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rtsp_url: rtspUrl, ocr_backend: ocrBackend }),
    })
    if (!resp.ok) { alert('Could not connect: ' + (await resp.text())); return }
    const data = await resp.json()
    setSession({ mode: 'live', sessionId: data.session_id, whepUrl: data.whep_url, mjpegUrl: data.mjpeg_url, rtspUrl, ocrBackend })
    setIncidents({})
  }

  const handleSelectFile = async (file, preprocessMode = 'none', ocrBackend = 'default') => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('preprocess_mode', preprocessMode)
    fd.append('ocr_backend', ocrBackend)
    const resp = await fetch('/monitor/upload', { method: 'POST', body: fd })
    if (!resp.ok) { alert('Upload failed'); return }
    const data = await resp.json()
    setSession({
      mode: 'upload',
      sessionId: data.session_id,
      videoUrl: data.video_url,
      preprocessMode: data.preprocess_mode,
      ocrBackend: data.ocr_backend,
      file,
    })
    setIncidents({})
  }

  const handleMarkLive = async () => {
    try {
      const id = await postMark(session.sessionId, { mode: 'live' })
      setIncidents((p) => (
        p[id] ? p : { ...p, [id]: {
          id, status: 'pending', markedAt: new Date().toISOString(),
          windowStartSec: 0, windowEndSec: 0, vehicles: {},
        }}
      ))
    } catch (e) { alert('Mark failed: ' + e.message) }
  }

  const handleMarkUpload = async (tStart, tEnd) => {
    try {
      const id = await postMark(session.sessionId, { mode: 'upload', t_start: tStart, t_end: tEnd })
      setIncidents((p) => (
        p[id] ? p : { ...p, [id]: {
          id, status: 'pending', markedAt: new Date().toISOString(),
          windowStartSec: tStart, windowEndSec: tEnd, vehicles: {},
        }}
      ))
    } catch (e) { alert('Mark failed: ' + e.message) }
  }

  // ── Layout ─────────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5">
        <SourceSelector onConnectLive={handleConnectLive} onSelectFile={handleSelectFile} />
      </div>
    )
  }

  return (
    <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5 flex gap-4">
      <div className="flex-1 min-w-0">
        {session.mode === 'live' ? (
          <>
            <LiveViewer whepUrl={session.whepUrl} mjpegUrl={session.mjpegUrl} />
            <div className="mt-3">
              <button
                onClick={handleMarkLive}
                className="text-sm px-5 py-3 rounded-lg bg-red-600 hover:bg-red-500
                           text-white font-bold w-full"
              >
                🚩 Mark Now (10s)
              </button>
            </div>
          </>
        ) : (
          <UploadViewer videoUrl={session.videoUrl} onMark={handleMarkUpload} />
        )}
      </div>
      <IncidentsPanel incidents={incidents} />
    </div>
  )
}
