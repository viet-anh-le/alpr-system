import { useEffect, useState } from 'react'

import { apiJson } from '../apiClient'
import TrackBufferModal from './TrackBufferModal'
import { Badge, Button, Drawer, EmptyState, Skeleton, cx } from './ui'
import { VEHICLE_LABEL } from './workbench/constants'

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
      description="Các phiên đã lưu, ảnh cắt đối chiếu và độ tin cậy theo tài khoản."
    >
      <div className="grid min-h-full lg:grid-cols-[320px_1fr]">
        <aside className="border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)] lg:border-b-0 lg:border-r">
          <div className="panel-header">
            <div>
              <p className="section-label">Phiên xử lý</p>
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
              <EmptyState title="Chưa có phiên">
                Sau khi xử lý video thành công, phiên và bản ghi nhận dạng sẽ xuất hiện ở đây.
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
                        {getJobStatusLabel(job.status)}
                      </Badge>
                    </div>
                    <p className="mt-2 data-font truncate text-[11px] text-[var(--color-text-subtle)]">#{job.session_id}</p>
                    <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                      {new Date(job.created_at).toLocaleString('vi')} · {job.total_records || 0} bản ghi
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
            <EmptyState title="Chọn một phiên">
              Danh sách bản ghi sẽ hiển thị ảnh cắt phương tiện, ảnh cắt biển số và độ tin cậy OCR.
            </EmptyState>
          ) : vehicles.length === 0 ? (
            <EmptyState title="Phiên chưa có bản ghi">
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
  const [bufferVehicle, setBufferVehicle] = useState(null)
  const confidence = Math.round((vehicle.plate_text_confidence || 0) * 100)
  const identity = formatRecognitionIdentity(vehicle)
  const clusters = Array.isArray(vehicle.clusters) ? vehicle.clusters : []

  return (
    <>
      <article className="overflow-hidden rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)]">
        <div className="grid grid-cols-2 gap-px bg-[var(--color-border)]">
          <HistoryImage src={vehicle.vehicle_thumbnail_url} alt="Ảnh phương tiện đối chiếu" />
          <HistoryImage src={vehicle.best_plate_frame?.image_url} alt="Ảnh biển số đối chiếu" dark />
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
            {identity} · {VEHICLE_LABEL[vehicle.vehicle_class] || vehicle.vehicle_class || 'Phương tiện'} · {formatOcrMethod(vehicle.ocr_method)}
          </p>
          <p className="data-font text-[11px] text-[var(--color-text-subtle)]">
            Khung {vehicle.first_seen_frame ?? '—'} → {vehicle.last_seen_frame ?? '—'}
          </p>
          <Button size="sm" variant="secondary" onClick={() => setBufferVehicle(toModalVehicle(vehicle))}>
            Bộ đệm theo vết
          </Button>

          {clusters.length > 1 && (
            <div className="space-y-2 border-t border-[var(--color-border)] pt-3">
              <p className="section-label">Cụm OCR đã lưu</p>
              {clusters.map((cluster) => {
                const clusterConfidence = Math.round((cluster.plate_text_confidence || 0) * 100)
                return (
                  <div
                    key={cluster.cluster_index}
                    className="rounded-lg border border-[var(--color-border)] bg-black/15 p-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <Badge tone="neutral">Cụm {(cluster.cluster_index ?? 0) + 1}</Badge>
                      <span className="text-[10px] text-[var(--color-text-subtle)]">
                        {cluster.frame_count || cluster.track_buffer?.length || 0} khung · {clusterConfidence}%
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-3">
                      <div className="h-14 w-24 flex-shrink-0 overflow-hidden rounded bg-black">
                        {cluster.best_plate_frame?.image_url ? (
                          <img
                            src={cluster.best_plate_frame.image_url}
                            alt={`Cụm ${(cluster.cluster_index ?? 0) + 1}`}
                            className="h-full w-full object-contain"
                          />
                        ) : (
                          <span className="flex h-full items-center justify-center text-[10px] text-[var(--color-text-subtle)]">
                            Không có ảnh
                          </span>
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="plate-font truncate text-sm font-bold tracking-widest">
                          {displayPlateText(cluster.plate_text) || '—'}
                        </p>
                        <Button
                          className="mt-2"
                          size="sm"
                          variant="secondary"
                          onClick={() => setBufferVehicle(toModalVehicle(vehicle, cluster))}
                        >
                          Xem bộ đệm
                        </Button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </article>

      {bufferVehicle && (
        <TrackBufferModal
          vehicle={bufferVehicle}
          jobId={vehicle.session_id}
          onClose={() => setBufferVehicle(null)}
        />
      )}
    </>
  )
}

function toModalVehicle(record, cluster = null) {
  if (!cluster) {
    return {
      ...record,
      id: record.track_id,
      recognition_id: record.track_id,
      cls: record.vehicle_class,
      plate: record.plate_text,
      chars: record.chars || [],
      vote_summary: record.ocr_vote_summary,
      vehicle_b64: record.vehicle_thumbnail_url,
      plate_b64: record.best_plate_frame?.image_url,
    }
  }

  return {
    ...record,
    ...cluster,
    id: record.track_id,
    recognition_id: record.track_id,
    track_id: record.track_id,
    vehicle_track_id: record.vehicle_track_id,
    plate_track_id: record.plate_track_id,
    cls: record.vehicle_class,
    plate: cluster.plate_text,
    chars: cluster.chars || [],
    vote_summary: cluster.ocr_vote_summary,
    vehicle_b64: record.vehicle_thumbnail_url,
    plate_b64: cluster.best_plate_frame?.image_url,
  }
}

function formatRecognitionIdentity(vehicle) {
  const parts = [`Kết quả #${vehicle.track_id}`]
  if (vehicle.vehicle_track_id !== undefined && vehicle.vehicle_track_id !== null) {
    parts.push(`Xe #${vehicle.vehicle_track_id}`)
  }
  if (vehicle.plate_track_id !== undefined && vehicle.plate_track_id !== null) {
    parts.push(`Biển số #${vehicle.plate_track_id}`)
  }
  return parts.join(' · ')
}

function getJobStatusLabel(status) {
  if (status === 'completed') return 'Hoàn tất'
  if (status === 'failed') return 'Có lỗi'
  if (status === 'processing') return 'Đang xử lý'
  return 'Đang chờ'
}

function formatOcrMethod(value) {
  if (!value) return 'OCR'
  const labels = {
    realtime_buffer: 'Bộ đệm thời gian thực',
    default: 'SmallLPR-Line-CTC (mặc định)',
    smalllpr_line_ctc: 'SmallLPR-Line-CTC',
    vietnamese_yolov5: 'YOLOv5 Việt Nam',
  }
  return labels[value] || value.replaceAll('_', ' ')
}

function HistoryImage({ src, alt, dark = false }) {
  return (
    <div className={cx('flex h-32 items-center justify-center', dark ? 'bg-black' : 'bg-black/30')}>
      {src ? (
        <img src={src} alt={alt} className="max-h-full max-w-full object-contain" />
      ) : (
        <span className="text-xs text-[var(--color-text-subtle)]">Không có ảnh</span>
      )}
    </div>
  )
}
