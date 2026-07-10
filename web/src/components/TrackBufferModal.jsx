import { useEffect, useState } from 'react'

import { apiFetch } from '../apiClient'
import { Badge, Dialog, EmptyState, Skeleton, cx } from './ui'
import {
  buildInlineRecord,
  frameScore,
  getTrackRecordId,
  mergePersistedRecord,
} from './trackBufferRecord'

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function TrackBufferModal({ vehicle, jobId, onClose }) {
  const initialRecord = buildInlineRecord(vehicle, jobId)
  const [record, setRecord] = useState(initialRecord)
  const [loading, setLoading] = useState(Boolean(vehicle && !initialRecord && jobId))
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!vehicle) return
    const inlineRecord = buildInlineRecord(vehicle, jobId)
    const trackId = getTrackRecordId(vehicle)
    setRecord(inlineRecord)
    setError(null)

    if (!jobId) {
      setLoading(false)
      if (!inlineRecord) setError('Không có dữ liệu bộ đệm cho kết quả này.')
      return
    }

    setLoading(!inlineRecord)

    apiFetch(`/records/${jobId}/${trackId}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(response.status === 404 ? 'Dữ liệu chưa được lưu xong, thử lại sau.' : `Lỗi HTTP ${response.status}`)
        }
        return response.json()
      })
      .then((data) => setRecord(mergePersistedRecord(data, inlineRecord)))
      .catch((err) => {
        if (!inlineRecord) setError(err.message)
      })
      .finally(() => setLoading(false))
  }, [jobId, vehicle])

  const frames = record?.track_buffer ?? []
  const sortedFrames = [...frames].sort((a, b) => frameScore(b) - frameScore(a))
  const bestIdx = record?.best_plate_frame?.frame_index
  const votes = Object.entries(record?.ocr_vote_summary ?? {}).sort((a, b) => b[1] - a[1])
  const titlePlate = displayPlateText(record?.plate_text || vehicle?.plate)
  const titleTrackId = vehicle?.recognition_id ?? vehicle?.id ?? vehicle?.track_id

  return (
    <Dialog
      open={!!vehicle}
      onClose={onClose}
      title={`Kết quả #${titleTrackId}${titlePlate ? ` · ${titlePlate}` : ''}`}
      description={record ? `${frames.length} ảnh đối chiếu · ${formatOcrMethod(record.ocr_method)} · ${Math.round((record.plate_text_confidence || 0) * 100)}% độ tin cậy` : 'Bộ đệm theo vết được lưu sau khi OCR hoàn tất.'}
      className="max-w-4xl"
    >
      <div className="max-h-[76vh] overflow-y-auto p-4">
        {loading && (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <Skeleton className="h-32" />
              <Skeleton className="h-32" />
            </div>
            <Skeleton className="h-16" />
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-5">
              {Array.from({ length: 10 }).map((_, index) => <Skeleton key={index} className="h-24" />)}
            </div>
          </div>
        )}

        {error && !loading && <EmptyState title="Không tải được bộ đệm theo vết">{error}</EmptyState>}

        {record && (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <ImageBox url={record.vehicle_thumbnail_url} label="Ảnh đại diện phương tiện" />
              <ImageBox
                url={record.best_plate_frame?.image_url || record.best_plate_frame?.image_b64}
                label={`Biển số rõ nhất · điểm ${frameScore(record.best_plate_frame).toFixed(2)}`}
                highlight
              />
            </div>

            {votes.length > 0 && (
              <section className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
                <p className="section-label">Tổng hợp phiếu OCR</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {votes.map(([text, count]) => (
                    <Badge
                      key={text}
                      tone={displayPlateText(text) === displayPlateText(record.plate_text) ? 'success' : 'neutral'}
                    >
                      <span className="plate-font">{displayPlateText(text)}</span>
                      <span className="text-[var(--color-text-subtle)]">×{count}</span>
                    </Badge>
                  ))}
                </div>
              </section>
            )}

            <section>
              <div className="mb-3 flex items-center justify-between">
                <p className="section-label">Bộ đệm theo vết · sắp xếp theo điểm</p>
                <Badge tone="info">{sortedFrames.length} khung</Badge>
              </div>
              {sortedFrames.length === 0 ? (
                <EmptyState title="Không có ảnh trong bộ đệm" />
              ) : (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-5">
                  {sortedFrames.map((frame, index) => (
                    <FrameCell
                      key={`${frame.frame_index}-${index}`}
                      frame={frame}
                      isBest={frame.frame_index === bestIdx}
                    />
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </Dialog>
  )
}

function normalizeImageSrc(src) {
  if (!src) return null
  return src.startsWith?.('http') || src.startsWith?.('data:')
    ? src
    : `data:image/jpeg;base64,${src}`
}

function ImageBox({ url, label, highlight = false }) {
  const src = normalizeImageSrc(url)
  return (
    <div className={cx('overflow-hidden rounded-[var(--radius-panel)] border bg-black', highlight ? 'border-emerald-300/45' : 'border-[var(--color-border)]')}>
      <div className="flex h-36 items-center justify-center">
        {src ? (
          <img src={src} alt={label} className="max-h-full max-w-full object-contain" />
        ) : (
          <span className="text-xs text-[var(--color-text-subtle)]">Không có ảnh</span>
        )}
      </div>
      <p className="border-t border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 py-2 text-xs font-semibold text-[var(--color-text-muted)]">
        {label}
      </p>
    </div>
  )
}

function FrameCell({ frame, isBest }) {
  const score = frameScore(frame)
  const normalizedSrc = normalizeImageSrc(frame.image_url || frame.image_b64)

  return (
    <div className={cx('overflow-hidden rounded-lg border bg-black', isBest ? 'border-emerald-300/55' : 'border-[var(--color-border)]')}>
      <div className="relative flex h-20 items-center justify-center">
        {normalizedSrc ? (
          <img src={normalizedSrc} alt={`khung ${frame.frame_index}`} className="max-h-full max-w-full object-contain" />
        ) : (
          <span className="text-[10px] text-[var(--color-text-subtle)]">không có ảnh</span>
        )}
        {isBest && <span className="absolute right-1 top-1 rounded bg-emerald-300 px-1.5 py-0.5 text-[9px] font-bold text-black">TỐT NHẤT</span>}
      </div>
      <div className="h-1 bg-white/10">
        <div className={cx('h-full', qualityColor(score))} style={{ width: `${Math.min(score * 100, 100)}%` }} />
      </div>
      <p className={cx('data-font px-1 py-1 text-center text-[10px]', qualityText(score))}>
        #{frame.frame_index} · {score.toFixed(2)}
      </p>
    </div>
  )
}

function qualityColor(quality) {
  if (quality >= 0.8) return 'bg-emerald-300'
  if (quality >= 0.6) return 'bg-amber-300'
  return 'bg-red-300'
}

function qualityText(quality) {
  if (quality >= 0.8) return 'text-emerald-100'
  if (quality >= 0.6) return 'text-amber-100'
  return 'text-red-100'
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
