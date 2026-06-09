import { useState, useCallback } from 'react'

import DropZone       from './components/DropZone'
import LiveFrame      from './components/LiveFrame'
import VehiclePanel   from './components/VehiclePanel'
import OcrStatsPanel  from './components/OcrStatsPanel'
import ErrorToast     from './components/ErrorToast'
import HistoryModal   from './components/HistoryModal'
import MonitorPage    from './components/monitor/MonitorPage'
import { useUpload }  from './hooks/useUpload'
import { useStream }  from './hooks/useStream'

export default function App() {
  const [mode,      setMode]      = useState('process')  // 'process' | 'monitor'
  const [vehicles,  setVehicles]  = useState({})
  const [status,    setStatus]    = useState('idle')
  const [progress,  setProgress]  = useState({ frame: 0, total: 0, pct: 0 })
  const [error,     setError]     = useState(null)
  const [jobId,     setJobId]     = useState(null)
  const [videoUrl,  setVideoUrl]  = useState(null)   // local object URL for preview
  const [frameB64,  setFrameB64]  = useState(null)   // latest annotated frame from SSE
  const [showHistory, setShowHistory] = useState(false)
  const [rejectedVehicles, setRejectedVehicles] = useState({})
  const [preprocessMode, setPreprocessMode] = useState('none')
  const [ocrBackend, setOcrBackend] = useState('default')

  const { uploadVideo } = useUpload()

  // ── SSE callbacks ─────────────────────────────────────────────────────────
  const handleVehicle  = useCallback((data) =>
    setVehicles(prev => ({ ...prev, [data.id]: data })), [])

  const handleProgress = useCallback((data) =>
    setProgress({ frame: data.frame, total: data.total, pct: data.pct }), [])

  const handleComplete = useCallback(() => {
    setStatus('done')
    setProgress(p => ({ ...p, pct: 100 }))
  }, [])

  const handleError = useCallback((msg) => {
    setError(msg)
    setStatus('error')
  }, [])

  const handleFrame = useCallback((data) => setFrameB64(data.b64), [])

  const handleRejectedVehicle = useCallback((data) =>
    setRejectedVehicles(prev => ({ ...prev, [data.id]: data })), [])

  useStream(jobId, {
    onVehicle:  handleVehicle,
    onRejectedVehicle: handleRejectedVehicle,
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError:    handleError,
    onFrame:    handleFrame,
  })

  // ── Actions ───────────────────────────────────────────────────────────────
  const handleFileSelect = async (file) => {
    // Create local video URL for in-browser preview
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoUrl(URL.createObjectURL(file))

    setStatus('uploading')
    setVehicles({})
    setRejectedVehicles({})
    setProgress({ frame: 0, total: 0, pct: 0 })
    setError(null)
    setJobId(null)

    try {
      const id = await uploadVideo(file, preprocessMode, ocrBackend)
      setJobId(id)
      setStatus('processing')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  const handleReset = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setStatus('idle')
    setVehicles({})
    setRejectedVehicles({})
    setProgress({ frame: 0, total: 0, pct: 0 })
    setError(null)
    setJobId(null)
    setVideoUrl(null)
    setFrameB64(null)
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  const vehicleList       = Object.values(vehicles).sort((a, b) => a.id - b.id)
  const rejectedList      = Object.values(rejectedVehicles).sort((a, b) => a.id - b.id)
  const totalDone         = vehicleList.filter(v => v.done).length
  const isIdle            = status === 'idle'

  return (
    <div className="bg-slate-900 min-h-screen flex flex-col text-white">
      {/* ── Header ── */}
      <header className="bg-slate-950 border-b border-slate-800 flex-shrink-0">
        <div className="max-w-screen-xl mx-auto px-5 py-3 flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={2}>
              <rect x="2" y="7" width="20" height="14" rx="2"
                    stroke="currentColor" fill="none" />
              <circle cx="12" cy="13" r="3" fill="currentColor" opacity={0.6} />
              <circle cx="12" cy="13" r="1.5" fill="currentColor" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-bold text-white leading-tight">
              ALPR — Nhận dạng Biển số Xe Việt Nam
            </h1>
            <p className="text-[10px] text-slate-400">
              Automatic License Plate Recognition · AI-powered
            </p>
          </div>

          <div className="ml-6 flex items-center gap-1">
            <button
              onClick={() => setMode('process')}
              className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
                mode === 'process' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Xử lý video
            </button>
            <button
              onClick={() => setMode('monitor')}
              className={`text-xs px-3 py-1.5 rounded-md transition-colors ${
                mode === 'monitor' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Giám sát sự cố
            </button>
          </div>

          {/* Status indicator (top-right) */}
          <div className="ml-auto flex items-center gap-2">
            {!isIdle && (
              <span className={`text-xs px-2.5 py-1 rounded-full font-medium
                ${status === 'done'       ? 'bg-emerald-900/50 text-emerald-400'
                : status === 'error'      ? 'bg-red-900/50 text-red-400'
                : 'bg-blue-900/50 text-blue-300'}`}>
                {status === 'done'       ? `✓ Hoàn tất · ${vehicleList.length} xe`
                 : status === 'error'    ? '✗ Lỗi'
                 : status === 'uploading'? '↑ Đang tải…'
                 : `⟳ Đang xử lý · ${progress.pct}%`}
              </span>
            )}
            {!isIdle && (
              <button
                onClick={handleReset}
                className="text-xs text-slate-400 hover:text-white border border-slate-700
                           hover:border-slate-500 px-3 py-1 rounded-lg transition-colors"
              >
                Video mới
              </button>
            )}
            <button
              onClick={() => setShowHistory(true)}
              className="text-xs ml-2 bg-slate-800 text-slate-300 hover:text-white border border-slate-700
                         hover:border-slate-500 px-3 py-1 rounded-lg transition-colors flex items-center gap-1.5"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Lịch sử
            </button>
          </div>
        </div>
      </header>

      {/* ── Monitor mode ── */}
      {mode === 'monitor' && <MonitorPage />}

      {/* ── Process mode: Main 2-column layout ── */}
      {mode === 'process' && (
      <div className="flex-1 max-w-screen-xl mx-auto w-full px-5 py-5
                      flex gap-4 items-start">

        {/* ── LEFT: Video / Drop zone (65%) ── */}
        <div className="flex-1 min-w-0">
          {isIdle ? (
            /* Drop zone shown when idle */
            <DropZone
              onFileSelect={handleFileSelect}
              dark
              preprocessMode={preprocessMode}
              onPreprocessModeChange={setPreprocessMode}
              ocrBackend={ocrBackend}
              onOcrBackendChange={setOcrBackend}
            />
          ) : (
            <>
              {/* Live annotated frame during processing, original video when done */}
              <LiveFrame
                frameB64={frameB64}
                videoUrl={videoUrl}
                progress={progress}
                status={status}
              />

              {/* OCR Statistics panel */}
              <OcrStatsPanel
                vehicles={vehicleList}
                rejectedVehicles={rejectedList}
                jobId={jobId}
              />
            </>
          )}
        </div>

        {/* ── RIGHT: Vehicle detection panel (35%) ── */}
        <div className="w-80 flex-shrink-0" style={{ height: 'calc(100vh - 72px)' }}>
          <VehiclePanel vehicles={vehicleList} totalDone={totalDone} jobId={jobId} />
        </div>
      </div>
      )}

      {/* Error toast */}
      {error && (
        <ErrorToast message={error} onDismiss={() => setError(null)} />
      )}

      {/* History Modal */}
      {showHistory && (
        <HistoryModal onClose={() => setShowHistory(false)} />
      )}
    </div>
  )
}
