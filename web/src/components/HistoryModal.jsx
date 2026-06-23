import { useEffect, useState } from 'react'

import { apiJson } from '../apiClient'
import { Badge, Drawer, EmptyState, Skeleton, cx } from './ui'

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function HistoryModal({ open, onClose }) {
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [vehicles, setVehicles] = useState([])
  const [loadingJobs, setLoadingJobs] = useState(false)
  const [loadingVehicles, setLoadingVehicles] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!open) return
    async function fetchJobs() {
      setLoadingJobs(true)
      setError(null)
      try {
        const data = await apiJson('/sessions?limit=50')
        const items = data.items || []
        setJobs(items)
        setSelectedJobId(items[0]?.session_id || null)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoadingJobs(false)
      }
    }
    fetchJobs()
  }, [open])

  useEffect(() => {
    if (!open || !selectedJobId) return
    async function fetchVehicles() {
      setVehicles([])
      setLoadingVehicles(true)
      setError(null)
      try {
        const data = await apiJson(`/sessions/${selectedJobId}/records`)
        setVehicles(data.items || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoadingVehicles(false)
      }
    }
    fetchVehicles()
  }, [open, selectedJobId])

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="Lịch sử nhận dạng"
      description="Các phiên đã lưu, crop chứng cứ và confidence theo tài khoản."
    >
      <div className="grid min-h-full lg:grid-cols-[320px_1fr]">
        <aside className="border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] lg:border-b-0 lg:border-r">
          <div className="panel-header">
            <div>
              <p className="section-label">Sessions</p>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{jobs.length} phiên gần nhất</p>
            </div>
          </div>
          <div className="max-h-[42vh] overflow-y-auto p-3 lg:max-h-none">
            {loadingJobs ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, index) => (
                  <Skeleton key={index} className="h-20" />
                ))}
              </div>
            ) : jobs.length === 0 ? (
              <EmptyState title="Chưa có session">
                Sau khi xử lý video thành công, session và recognition records sẽ xuất hiện ở đây.
              </EmptyState>
            ) : (
              <div className="space-y-2">
                {jobs.map((job) => (
                  <button
                    key={job.session_id}
                    type="button"
                    onClick={() => setSelectedJobId(job.session_id)}
                    className={cx(
                      'w-full rounded-xl border p-3 text-left transition-colors duration-200',
                      selectedJobId === job.session_id
                        ? 'border-cyan-300/45 bg-cyan-300/10'
                        : 'border-[var(--color-border)] bg-black/10 hover:bg-white/5',
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-sm font-semibold">{job.source_filename}</p>
                      <Badge tone={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'danger' : 'info'}>
                        {job.status}
                      </Badge>
                    </div>
                    <p className="mt-2 data-font truncate text-[11px] text-[var(--color-text-subtle)]">#{job.session_id}</p>
                    <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                      {new Date(job.created_at).toLocaleString('vi')} · {job.total_records || 0} records
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </aside>

        <section className="min-h-0 bg-[var(--color-bg)] p-4">
          {error ? (
            <EmptyState title="Không tải được lịch sử">{error}</EmptyState>
          ) : loadingVehicles ? (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {Array.from({ length: 6 }).map((_, index) => (
                <Skeleton key={index} className="h-64" />
              ))}
            </div>
          ) : !selectedJobId ? (
            <EmptyState title="Chọn một session">
              Danh sách record sẽ hiển thị crop phương tiện, crop biển số và confidence OCR.
            </EmptyState>
          ) : vehicles.length === 0 ? (
            <EmptyState title="Session chưa có record">
              Không tìm thấy biển số hợp lệ trong phiên này hoặc dữ liệu chưa được lưu.
            </EmptyState>
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {vehicles.map((vehicle) => (
                <HistoryRecord key={`${vehicle.session_id}-${vehicle.track_id}`} vehicle={vehicle} />
              ))}
            </div>
          )}
        </section>
      </div>
    </Drawer>
  )
}

function HistoryRecord({ vehicle }) {
  const confidence = Math.round((vehicle.plate_text_confidence || 0) * 100)
  const identity = formatRecognitionIdentity(vehicle)
  return (
    <article className="overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
      <div className="grid grid-cols-2 gap-px bg-[var(--color-border)]">
        <HistoryImage src={vehicle.vehicle_thumbnail_url} alt="Vehicle evidence" />
        <HistoryImage src={vehicle.best_plate_frame?.image_url} alt="Plate evidence" dark />
      </div>
      <div className="space-y-3 p-3">
        <div className="flex items-start justify-between gap-3">
          <p className="plate-font min-w-0 truncate text-lg font-bold tracking-widest">
            {displayPlateText(vehicle.plate_text) || '—'}
          </p>
          <Badge tone={confidence >= 90 ? 'success' : confidence >= 70 ? 'warning' : 'danger'}>
            {confidence}%
          </Badge>
        </div>
        <p className="text-xs text-[var(--color-text-muted)]">
          {identity} · {vehicle.vehicle_class || 'vehicle'} · {vehicle.ocr_method || 'OCR'}
        </p>
        <p className="data-font text-[11px] text-[var(--color-text-subtle)]">
          Frame {vehicle.first_seen_frame ?? '—'} → {vehicle.last_seen_frame ?? '—'}
        </p>
      </div>
    </article>
  )
}

function formatRecognitionIdentity(vehicle) {
  const parts = [`Result #${vehicle.track_id}`]
  if (vehicle.vehicle_track_id !== undefined && vehicle.vehicle_track_id !== null) {
    parts.push(`Vehicle #${vehicle.vehicle_track_id}`)
  }
  if (vehicle.plate_track_id !== undefined && vehicle.plate_track_id !== null) {
    parts.push(`Plate #${vehicle.plate_track_id}`)
  }
  return parts.join(' · ')
}

function HistoryImage({ src, alt, dark = false }) {
  return (
    <div className={cx('flex h-32 items-center justify-center', dark ? 'bg-black' : 'bg-black/30')}>
      {src ? (
        <img src={src} alt={alt} className="max-h-full max-w-full object-contain" />
      ) : (
        <span className="text-xs text-[var(--color-text-subtle)]">No image</span>
      )}
    </div>
  )
}
