import { Link } from 'react-router-dom'

import { useAuth } from '../auth'
import { Badge, Button } from '../components/ui'

const pipelineSteps = [
  ['Source', 'Upload video hoặc RTSP monitor window'],
  ['Detection', 'YOLO vehicle + YOLOv8 OBB plate detection'],
  ['Tracking', 'BoT-SORT/ReID, association qua nhiều frame'],
  ['OCR', 'SmallLPR/PARSeq/YOLOv5 backends + voting'],
  ['Evidence', 'Crops, confidence, rejected candidates, history'],
]

const metrics = [
  ['YOLOv8 OBB mAP50', '98.29%', 'artifact detector biển số'],
  ['SmallLPR-NAR val acc', '95.811%', 'OCR validation nội bộ'],
  ['PARSeq seq acc', '95.459%', 'OCR validation nội bộ'],
  ['Quality router binary', '95.603%', 'lọc crop phù hợp OCR'],
  ['Async pipeline speedup', '1.43x-1.81x', 'video benchmark'],
]

const surfaces = [
  ['Processing workbench', 'Upload video, theo dõi tiến trình SSE, review plate evidence và confidence.'],
  ['Event monitor', 'Kết nối RTSP hoặc video dài, mark cửa sổ ngắn để phân tích nhanh.'],
  ['History review', 'Truy xuất session, recognition records, crop phương tiện và crop biển số đã lưu.'],
]

function getPrimaryCta(user, loading) {
  if (loading) return { to: '/login', label: 'Đang kiểm tra phiên' }
  if (user) return { to: '/dashboard', label: 'Mở workbench' }
  return { to: '/register', label: 'Tạo tài khoản demo' }
}

export default function LandingPage() {
  const { user, loading } = useAuth()
  const primaryCta = getPrimaryCta(user, loading)

  return (
    <main className="app-shell min-h-screen">
      <header className="app-topbar">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
          <Link to="/" className="flex min-w-0 items-center gap-3">
            <span className="brand-mark"><PlateGlyph /></span>
            <span className="truncate text-sm font-bold sm:text-base">ALPR Vietnamese</span>
          </Link>
          <nav className="ml-auto hidden items-center gap-5 text-sm font-semibold text-[var(--color-text-muted)] md:flex">
            <a href="#pipeline" className="hover:text-[var(--color-text)]">Pipeline</a>
            <a href="#evidence" className="hover:text-[var(--color-text)]">Evidence</a>
            <a href="#metrics" className="hover:text-[var(--color-text)]">Metrics</a>
          </nav>
          {!user && !loading && (
            <Link to="/login" className="hidden text-sm font-semibold text-[var(--color-text-muted)] hover:text-[var(--color-text)] sm:inline">
              Đăng nhập
            </Link>
          )}
          <Link to={primaryCta.to}>
            <Button size="sm" variant="primary">{user ? 'Dashboard' : 'Demo'}</Button>
          </Link>
        </div>
      </header>

      <section className="mx-auto grid max-w-7xl gap-8 px-4 py-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:py-20">
        <div>
          <Badge tone="info">Vietnamese ALPR · thesis-grade computer vision</Badge>
          <h1 className="mt-6 max-w-4xl text-4xl font-bold leading-tight text-[var(--color-text)] sm:text-5xl">
            Evidence workbench for Vietnamese license plate recognition
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-[var(--color-text-muted)]">
            Upload video or monitor a short event window, run the ALPR pipeline, then inspect plate text, confidence, vehicle crops, plate crops, rejected candidates, and saved history.
          </p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <Link to={primaryCta.to}><Button variant="primary" size="lg">{primaryCta.label}</Button></Link>
            <a href="#pipeline"><Button variant="secondary" size="lg">Xem pipeline</Button></a>
          </div>
        </div>

        <HeroWorkbench />
      </section>

      <section id="pipeline" className="border-y border-[var(--color-border)] bg-[var(--color-bg-elevated)]/70">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
          <div className="mb-8 max-w-3xl">
            <p className="section-label">Pipeline</p>
            <h2 className="mt-3 text-2xl font-bold">Video-first, track-level ALPR</h2>
            <p className="mt-3 text-[var(--color-text-muted)]">
              The product narrative matches the actual repo architecture: source frames become tracked vehicles, plate crops, OCR votes, validation, and evidence records.
            </p>
          </div>
          <div className="grid gap-3 lg:grid-cols-5">
            {pipelineSteps.map(([title, text]) => (
              <article key={title} className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black/10 p-4">
                <p className="data-font text-xs font-bold text-cyan-100">{title}</p>
                <p className="mt-3 text-sm leading-6 text-[var(--color-text-muted)]">{text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="evidence" className="mx-auto grid max-w-7xl gap-6 px-4 py-14 sm:px-6 lg:grid-cols-[0.8fr_1.2fr] lg:items-start">
        <div>
          <p className="section-label">Evidence model</p>
          <h2 className="mt-3 text-2xl font-bold">The UI explains what the model knows</h2>
          <p className="mt-3 text-[var(--color-text-muted)]">
            A reviewer should not have to trust a single recognized string. The workbench exposes source media, result status, crop evidence, per-character confidence, OCR votes, rejected candidates, and history.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          {surfaces.map(([title, text]) => (
            <article key={title} className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
              <h3 className="text-base font-bold">{title}</h3>
              <p className="mt-3 text-sm leading-6 text-[var(--color-text-muted)]">{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="metrics" className="bg-[var(--color-bg-elevated)]/70">
        <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
          <div className="mb-8 max-w-3xl">
            <p className="section-label">Measured claims</p>
            <h2 className="mt-3 text-2xl font-bold">Metrics are presented as artifacts, not marketing promises</h2>
          </div>
          <div className="overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)]">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="bg-[var(--color-bg)] text-[var(--color-text-muted)]">
                <tr>
                  <th className="px-4 py-3 font-semibold">Artifact</th>
                  <th className="px-4 py-3 font-semibold">Value</th>
                  <th className="px-4 py-3 font-semibold">Context</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-surface)]">
                {metrics.map(([name, value, context]) => (
                  <tr key={name}>
                    <td className="px-4 py-3 font-semibold">{name}</td>
                    <td className="data-font px-4 py-3 text-cyan-100">{value}</td>
                    <td className="px-4 py-3 text-[var(--color-text-muted)]">{context}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
        <div className="surface-panel flex flex-col gap-5 p-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="section-label">Try the workbench</p>
            <h2 className="mt-2 text-2xl font-bold">Run a video and inspect the evidence trail.</h2>
          </div>
          <Link to={primaryCta.to}><Button variant="primary" size="lg">{primaryCta.label}</Button></Link>
        </div>
      </section>
    </main>
  )
}

function HeroWorkbench() {
  return (
    <div className="surface-panel overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Workbench preview</p>
          <p className="mt-1 text-sm font-semibold">job_2026_0613 · processing</p>
        </div>
        <Badge tone="success">4 plates</Badge>
      </div>
      <div className="grid gap-px bg-[var(--color-border)] md:grid-cols-[1.25fr_0.75fr]">
        <div className="bg-black p-4">
          <div className="relative aspect-video rounded-lg border border-cyan-300/30 bg-[var(--color-bg)] scanline-bg">
            <div className="absolute left-[18%] top-[24%] h-[42%] w-[58%] rounded border border-cyan-300" />
            <div className="absolute bottom-[26%] left-[34%] rounded border border-cyan-100 bg-cyan-300 px-3 py-1 plate-font text-xs font-bold tracking-widest text-black">
              29A-678.90
            </div>
            <div className="absolute bottom-3 left-3 rounded-lg bg-black/70 px-3 py-2 text-xs text-cyan-50">
              track #12 · OCR valid · 92%
            </div>
          </div>
        </div>
        <div className="space-y-3 bg-[var(--color-bg-elevated)] p-4">
          {['29A-678.90', '30A-123.45', '51F-888.88'].map((plate, index) => (
            <div key={plate} className="rounded-lg border border-[var(--color-border)] bg-black/15 p-3">
              <div className="flex items-center justify-between">
                <span className="plate-font text-sm font-bold tracking-wider">{plate}</span>
                <Badge tone={index === 0 ? 'success' : 'info'}>{index === 0 ? '92%' : 'review'}</Badge>
              </div>
              <p className="mt-2 text-xs text-[var(--color-text-muted)]">crop evidence · track buffer saved</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function PlateGlyph() {
  return (
    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="7" width="18" height="10" rx="2" />
      <path d="M7 11h4M14 11h3M7 14h10" strokeLinecap="round" />
    </svg>
  )
}
