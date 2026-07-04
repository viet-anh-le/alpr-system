import { useCallback, useRef, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useNavigate } from 'react-router-dom'

import HistoryModal from './components/HistoryModal'
import MonitorPage from './components/monitor/MonitorPage'
import { Badge, Button, SegmentedControl, TextInput, Toast } from './components/ui'
import { MediaStage, ResultsPanel, SourcePanel } from './components/workbench'
import { useAuth } from './auth'
import { useStream } from './hooks/useStream'
import { useUpload } from './hooks/useUpload'
import LandingPage from './pages/LandingPage'

function DashboardPage() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState('process')
  const [vehicles, setVehicles] = useState({})
  const [rejectedVehicles, setRejectedVehicles] = useState({})
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState({ frame: 0, total: 0, pct: 0 })
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null)
  const [processedVideoExpected, setProcessedVideoExpected] = useState(false)
  const [previewFrame, setPreviewFrame] = useState(null)
  const [preprocessedPreviewFrame, setPreprocessedPreviewFrame] = useState(null)
  const [showHistory, setShowHistory] = useState(false)
  const [preprocessMode, setPreprocessMode] = useState('none')
  const [ocrBackend, setOcrBackend] = useState('default')
  const [sourcePanelKey, setSourcePanelKey] = useState(0)
  const activeSessionRef = useRef(0)

  const { uploadVideo } = useUpload()

  const handleVehicle = useCallback((data) => {
    setVehicles((prev) => ({ ...prev, [data.id]: data }))
  }, [])

  const handleRejectedVehicle = useCallback((data) => {
    setRejectedVehicles((prev) => ({ ...prev, [data.id]: data }))
    setVehicles((prev) => {
      if (!prev[data.id]) return prev
      const next = { ...prev }
      delete next[data.id]
      return next
    })
  }, [])

  const handleProgress = useCallback((data) => {
    setProgress({ frame: data.frame, total: data.total, pct: data.pct })
  }, [])

  const handleComplete = useCallback((data) => {
    setProcessedVideoUrl(data?.processed_video_url || null)
    if (typeof data?.preprocess_mode === 'string') {
      setProcessedVideoExpected(data.preprocess_mode !== 'none')
    }
    setStatus('done')
    setProgress((prev) => ({ ...prev, pct: 100 }))
  }, [])

  const handleError = useCallback((message) => {
    setError(message)
    setStatus('error')
  }, [])

  const handleFrame = useCallback((data) => {
    setPreviewFrame(data?.b64 ? data : null)
  }, [])

  const handlePreprocessedFrame = useCallback((data) => {
    setPreprocessedPreviewFrame(data?.b64 ? data : null)
  }, [])

  useStream(jobId, {
    onVehicle: handleVehicle,
    onRejectedVehicle: handleRejectedVehicle,
    onProgress: handleProgress,
    onComplete: handleComplete,
    onError: handleError,
    onFrame: handleFrame,
    onPreprocessedFrame: handlePreprocessedFrame,
  })

  const handleFileSelect = async (file) => {
    const sessionId = activeSessionRef.current + 1
    activeSessionRef.current = sessionId

    if (videoUrl) URL.revokeObjectURL(videoUrl)
    setVideoUrl(URL.createObjectURL(file))
    setStatus('uploading')
    setVehicles({})
    setRejectedVehicles({})
    setProgress({ frame: 0, total: 0, pct: 0 })
    setError(null)
    setJobId(null)
    setPreviewFrame(null)
    setPreprocessedPreviewFrame(null)
    setProcessedVideoUrl(null)
    setProcessedVideoExpected(preprocessMode !== 'none')

    try {
      const upload = await uploadVideo(file, preprocessMode, ocrBackend)
      if (activeSessionRef.current !== sessionId) return
      setProcessedVideoExpected(upload.processedVideoExpected)
      setJobId(upload.jobId)
      setStatus('processing')
    } catch (err) {
      if (activeSessionRef.current !== sessionId) return
      setError(err.message)
      setStatus('error')
    }
  }

  const handleReset = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
    activeSessionRef.current += 1
    setSourcePanelKey((value) => value + 1)
    setStatus('idle')
    setVehicles({})
    setRejectedVehicles({})
    setProgress({ frame: 0, total: 0, pct: 0 })
    setError(null)
    setJobId(null)
    setVideoUrl(null)
    setProcessedVideoUrl(null)
    setProcessedVideoExpected(false)
    setPreviewFrame(null)
    setPreprocessedPreviewFrame(null)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  const vehicleList = Object.values(vehicles).sort((a, b) => a.id - b.id)
  const rejectedList = Object.values(rejectedVehicles).sort((a, b) => a.id - b.id)
  const totalDone = vehicleList.filter((vehicle) => vehicle.done).length

  return (
    <div className="app-shell">
      <AppTopbar
        user={user}
        mode={mode}
        onModeChange={setMode}
        status={status}
        progress={progress}
        vehicleCount={vehicleList.length}
        onReset={handleReset}
        onHistory={() => setShowHistory(true)}
        onLogout={handleLogout}
      />

      <main className="mx-auto w-full max-w-[1500px] px-4 pb-16 pt-5 sm:px-6">
        {mode === 'process' ? (
          <div className="evidence-grid">
            <div className="space-y-4">
              <SourcePanel
                key={sourcePanelKey}
                onFileSelect={handleFileSelect}
                preprocessMode={preprocessMode}
                onPreprocessModeChange={setPreprocessMode}
                ocrBackend={ocrBackend}
                onOcrBackendChange={setOcrBackend}
                disabled={status === 'uploading' || status === 'processing'}
                compact={status !== 'idle'}
              />
              <MediaStage
                previewFrame={previewFrame}
                preprocessedPreviewFrame={preprocessedPreviewFrame}
                videoUrl={videoUrl}
                processedVideoUrl={processedVideoUrl}
                processedVideoExpected={processedVideoExpected}
                preprocessMode={preprocessMode}
                progress={progress}
                status={status}
              />
            </div>
            <ResultsPanel
              vehicles={vehicleList}
              rejectedVehicles={rejectedList}
              totalDone={totalDone}
              jobId={jobId}
              status={status}
            />
          </div>
        ) : (
          <MonitorPage />
        )}
      </main>

      <Toast message={error} onDismiss={() => setError(null)} />
      <HistoryModal open={showHistory} onClose={() => setShowHistory(false)} />
    </div>
  )
}

function AppTopbar({
  user,
  mode,
  onModeChange,
  status,
  progress,
  vehicleCount,
  onReset,
  onHistory,
  onLogout,
}) {
  const statusBadge = getStatusBadge(status, progress, vehicleCount)

  return (
    <header className="app-topbar">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-3 px-4 py-3 sm:px-6 lg:flex-row lg:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <div className="brand-mark">
            <PlateGlyph />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold text-[var(--color-text)] sm:text-base">
              Bàn kiểm chứng ALPR Việt Nam
            </h1>
            <p className="truncate text-xs text-[var(--color-text-muted)]">
              Phát hiện · Theo vết · OCR · Kiểm chứng chứng cứ
            </p>
          </div>
        </div>

        <SegmentedControl
          value={mode}
          onChange={onModeChange}
          options={[
            { value: 'process', label: 'Xử lí toàn bộ' },
            { value: 'monitor', label: 'Xử lí trích đoạn' },
          ]}
          className="lg:ml-6"
        />

        <div className="flex flex-wrap items-center gap-2 lg:ml-auto">
          <Badge tone={statusBadge.tone}>{statusBadge.label}</Badge>
          {status !== 'idle' && (
            <Button size="sm" variant="secondary" onClick={onReset}>
              Nguồn mới
            </Button>
          )}
          <Button size="sm" variant="secondary" onClick={onHistory}>
            Lịch sử
          </Button>
          {user && <span className="hidden text-xs text-[var(--color-text-muted)] md:inline">{user.name}</span>}
          <Button size="sm" variant="ghost" onClick={onLogout}>
            Đăng xuất
          </Button>
        </div>
      </div>
    </header>
  )
}

function getStatusBadge(status, progress, vehicleCount) {
  if (status === 'done') return { label: `Hoàn tất · ${vehicleCount} biển số`, tone: 'success' }
  if (status === 'error') return { label: 'Có lỗi', tone: 'danger' }
  if (status === 'uploading') return { label: 'Đang tải lên', tone: 'info' }
  if (status === 'processing') return { label: `Đang phân tích · ${Math.round(progress.pct || 0)}%`, tone: 'info' }
  return { label: 'Sẵn sàng', tone: 'neutral' }
}

function AuthPage({ mode }) {
  const isRegister = mode === 'register'
  const { login, register } = useAuth()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setError(null)

    if (!email.trim() || !password) {
      setError('Vui lòng nhập email và mật khẩu.')
      return
    }
    if (isRegister && !name.trim()) {
      setError('Vui lòng nhập họ tên.')
      return
    }
    if (isRegister && password.length < 8) {
      setError('Mật khẩu cần ít nhất 8 ký tự.')
      return
    }

    setLoading(true)
    try {
      if (isRegister) {
        await register({ name: name.trim(), email: email.trim(), password })
      } else {
        await login({ email: email.trim(), password })
      }
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell flex min-h-screen items-center justify-center px-4 py-10">
      <div className="grid w-full max-w-5xl gap-5 lg:grid-cols-[1.05fr_0.95fr] lg:items-stretch">
        <section className="surface-panel overflow-hidden p-6 sm:p-8">
          <div className="brand-mark mb-6">
            <PlateGlyph />
          </div>
          <p className="section-label">Không gian bảo vệ</p>
          <h1 className="mt-3 max-w-xl text-3xl font-bold leading-tight text-[var(--color-text)]">
            {isRegister ? 'Tạo không gian kiểm chứng ALPR' : 'Đăng nhập bàn kiểm chứng ALPR'}
          </h1>
          <p className="mt-4 max-w-xl text-base leading-7 text-[var(--color-text-muted)]">
            Bảng điều khiển bảo vệ phiên xử lý, lịch sử nhận dạng, ảnh cắt chứng cứ và bộ đệm theo vết theo từng tài khoản.
          </p>
          <div className="mt-8 grid gap-3 sm:grid-cols-3">
            <AuthMetric value="SSE" label="Sự kiện thời gian thực" />
            <AuthMetric value="CSRF" label="Ghi dữ liệu có bảo vệ" />
            <AuthMetric value="DB" label="Lịch sử chứng cứ" />
          </div>
        </section>

        <section className="surface-panel overflow-hidden">
          <div className="panel-header">
            <div>
              <p className="section-label">{isRegister ? 'Đăng kí' : 'Đăng nhập'}</p>
              <h2 className="mt-1 text-lg font-bold">{isRegister ? 'Tạo tài khoản' : 'Mở bảng điều khiển'}</h2>
            </div>
            <Badge tone="info">Bản trình diễn tốt nghiệp</Badge>
          </div>
          <form onSubmit={submit} className="space-y-4 p-5">
            {isRegister && (
              <label className="block space-y-1.5">
                <span className="text-xs font-semibold text-[var(--color-text-muted)]">Họ tên</span>
                <TextInput value={name} onChange={(event) => setName(event.target.value)} autoComplete="name" />
              </label>
            )}
            <label className="block space-y-1.5">
              <span className="text-xs font-semibold text-[var(--color-text-muted)]">Email</span>
              <TextInput type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" />
            </label>
            <label className="block space-y-1.5">
              <span className="text-xs font-semibold text-[var(--color-text-muted)]">Mật khẩu</span>
              <TextInput
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={isRegister ? 'new-password' : 'current-password'}
              />
            </label>

            {error && (
              <div className="rounded-[var(--radius-control)] border border-red-300/30 bg-red-500/10 px-3 py-2 text-sm text-red-100">
                {error}
              </div>
            )}

            <Button type="submit" variant="primary" fullWidth loading={loading}>
              {isRegister ? 'Đăng kí' : 'Đăng nhập'}
            </Button>
            <Button
              fullWidth
              variant="ghost"
              onClick={() => navigate(isRegister ? '/login' : '/register')}
            >
              {isRegister ? 'Đã có tài khoản? Đăng nhập' : 'Chưa có tài khoản? Đăng kí'}
            </Button>
          </form>
        </section>
      </div>
    </main>
  )
}

function AuthMetric({ value, label }) {
  return (
    <div className="rounded-xl border border-[var(--color-border)] bg-black/15 p-3">
      <div className="data-font text-lg font-bold text-cyan-100">{value}</div>
      <div className="mt-1 text-xs text-[var(--color-text-muted)]">{label}</div>
    </div>
  )
}

function LoadingScreen() {
  return (
    <div className="app-shell flex min-h-screen items-center justify-center">
      <div className="surface-panel flex items-center gap-3 px-5 py-4 text-sm text-[var(--color-text-muted)]">
        <span className="h-4 w-4 rounded-full border-2 border-cyan-300 border-t-transparent animate-spin" />
        Đang kiểm tra phiên đăng nhập…
      </div>
    </div>
  )
}

function RequireAuth({ children }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingScreen />
  if (!user) return <Navigate to="/login" replace />
  return children
}

function PublicAuthRoute({ mode }) {
  const { user, loading } = useAuth()
  if (loading) return <LoadingScreen />
  if (user) return <Navigate to="/dashboard" replace />
  return <AuthPage mode={mode} />
}

function PlateGlyph() {
  return (
    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="7" width="18" height="10" rx="2" />
      <path d="M7 11h4M14 11h3M7 14h10" strokeLinecap="round" />
    </svg>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<PublicAuthRoute mode="login" />} />
        <Route path="/register" element={<PublicAuthRoute mode="register" />} />
        <Route
          path="/dashboard"
          element={
            <RequireAuth>
              <DashboardPage />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
