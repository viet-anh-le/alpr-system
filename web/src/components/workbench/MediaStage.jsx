import { Badge, EmptyState, Progress } from '../ui'

function statusCopy(status) {
  if (status === 'done') return ['Hoàn tất', 'success']
  if (status === 'error') return ['Có lỗi', 'danger']
  if (status === 'uploading') return ['Đang tải lên', 'info']
  if (status === 'processing') return ['Đang phân tích', 'info']
  return ['Sẵn sàng', 'neutral']
}

export default function MediaStage({ frameB64, videoUrl, progress, status }) {
  const isDone = status === 'done'
  const isError = status === 'error'
  const hasFrame = !!frameB64
  const [label, tone] = statusCopy(status)

  return (
    <section className="surface-panel overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Media evidence</p>
          <h2 className="mt-1 text-lg font-bold">Khung hình và annotation</h2>
        </div>
        <Badge tone={tone}>{label}</Badge>
      </div>

      <div className="p-4">
        <div className="relative overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-black scanline-bg">
          {hasFrame && !isDone && (
            <img
              src={`data:image/jpeg;base64,${frameB64}`}
              alt="Khung hình đã annotate"
              className="block max-h-[68vh] w-full object-contain"
              draggable={false}
            />
          )}

          {(isDone || !hasFrame) && videoUrl && (
            <video
              key={videoUrl}
              src={videoUrl}
              controls
              muted
              autoPlay={!isDone}
              loop={isDone}
              className="block max-h-[68vh] w-full object-contain"
            />
          )}

          {!videoUrl && !hasFrame && (
            <div className="p-4">
              <EmptyState title="Chưa có nguồn phân tích">
                Chọn video upload hoặc ghi camera clip để xem khung hình annotate và tiến trình OCR tại đây.
              </EmptyState>
            </div>
          )}

          {videoUrl && !isDone && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/70 to-transparent px-4 pb-4 pt-16">
              <div className="mb-2 flex items-center justify-between gap-3 text-sm text-white">
                <div className="flex items-center gap-2">
                  {!isDone && !isError && <span className="h-2 w-2 rounded-full bg-cyan-300" />}
                  <span className="font-semibold">{label}</span>
                </div>
                <span className="data-font font-bold">{Math.round(progress.pct || 0)}%</span>
              </div>
              <Progress value={progress.pct || 0} tone={isDone ? 'success' : isError ? 'danger' : 'info'} />
              {progress.frame > 0 && (
                <p className="mt-2 data-font text-xs text-white/70">
                  Frame {progress.frame.toLocaleString('vi')} / {progress.total.toLocaleString('vi')}
                </p>
              )}
            </div>
          )}
        </div>
        {videoUrl && isDone && (
          <div className="mt-3 rounded-[var(--radius-control)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-4 py-3">
            <div className="mb-2 flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold">{label} · original video ready for review</span>
              <span className="data-font font-bold">{Math.round(progress.pct || 0)}%</span>
            </div>
            <Progress value={progress.pct || 100} tone="success" />
          </div>
        )}
      </div>
    </section>
  )
}
