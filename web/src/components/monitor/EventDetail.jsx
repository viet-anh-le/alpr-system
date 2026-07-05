import { useState } from 'react'

import { Badge, Button, Dialog, EmptyState, cx } from '../ui'
import { VEHICLE_LABEL } from '../workbench/constants'

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function EventDetail({ event }) {
  const vehicles = Object.values(event.vehicles || {})
  if (vehicles.length === 0) {
    return <EmptyState title="Không có ảnh phương tiện" />
  }

  return (
    <div className="mt-3 space-y-3">
      {vehicles.map((vehicle) => (
        <EventVehicleDetail key={vehicle.track_id ?? vehicle.id} vehicle={vehicle} />
      ))}
    </div>
  )
}

function EventVehicleDetail({ vehicle }) {
  const [showBuffer, setShowBuffer] = useState(false)
  const frames = vehicle.track_buffer || []
  const vehicleId = vehicle.track_id ?? vehicle.id
  const identityLabel = formatRecognitionIdentity(vehicle)
  const confidence = Math.round((vehicle.confidence || 0) * 100)

  return (
    <article className="rounded-xl border border-[var(--color-border)] bg-black/15 p-3">
      <div className="grid grid-cols-2 gap-2">
        <ImageBox src={imageSrc(vehicle.vehicle_b64)} alt={`Xe ${vehicleId}`} fallback="Không có ảnh xe" />
        <ImageBox src={imageSrc(vehicle.plate_b64)} alt={displayPlateText(vehicle.plate)} fallback="Không có ảnh biển số" dark />
      </div>
      <div className="mt-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="plate-font truncate text-base font-bold tracking-widest text-emerald-100">
            {displayPlateText(vehicle.plate) || '—'}
          </p>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">
            {identityLabel} · {VEHICLE_LABEL[vehicle.cls] || vehicle.cls || 'Phương tiện'} · {vehicle.ocr_frames || 0} khung
          </p>
        </div>
        <Badge tone={confidence >= 90 ? 'success' : confidence >= 70 ? 'warning' : 'danger'}>
          {confidence || '—'}%
        </Badge>
      </div>
      <Button className="mt-3" size="sm" onClick={() => setShowBuffer(true)}>
        Bộ đệm ({frames.length})
      </Button>

      <TrackBufferDialog
        open={showBuffer}
        vehicle={vehicle}
        frames={frames}
        onClose={() => setShowBuffer(false)}
      />
    </article>
  )
}

function TrackBufferDialog({ open, vehicle, frames, onClose }) {
  const identityLabel = formatRecognitionIdentity(vehicle)
  const sortedFrames = [...frames].sort((a, b) => frameScore(b) - frameScore(a))

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Sự kiện ${identityLabel}`}
      description={`${sortedFrames.length} khung trong bộ đệm sự kiện`}
      className="max-w-3xl"
    >
      <div className="max-h-[70vh] overflow-y-auto p-4">
        {sortedFrames.length === 0 ? (
          <EmptyState title="Không có ảnh trong bộ đệm" />
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 md:grid-cols-5">
            {sortedFrames.map((frame, index) => (
              <FrameCell key={`${frame.frame_index}-${frame.candidate_method || ''}-${index}`} frame={frame} />
            ))}
          </div>
        )}
      </div>
    </Dialog>
  )
}

function FrameCell({ frame }) {
  const score = frameScore(frame)
  const src = imageSrc(frame.image_b64) || frame.image_url

  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-border)] bg-black">
      <div className="flex h-20 items-center justify-center">
        {src ? (
          <img src={src} alt={`khung ${frame.frame_index}`} className="max-h-full max-w-full object-contain" />
        ) : (
          <span className="text-[10px] text-[var(--color-text-subtle)]">không có ảnh</span>
        )}
      </div>
      <div className="h-1 bg-white/10">
        <div className={cx('h-full', qualityColor(score))} style={{ width: `${Math.min(score * 100, 100)}%` }} />
      </div>
      <p className="data-font px-1 py-1 text-center text-[10px] text-[var(--color-text-muted)]">
        #{frame.frame_index} · {score.toFixed(2)}
      </p>
    </div>
  )
}

function formatRecognitionIdentity(vehicle) {
  const resultId = vehicle.recognition_id ?? vehicle.track_id ?? vehicle.id
  const parts = [`Kết quả #${resultId}`]
  if (vehicle.vehicle_track_id !== undefined && vehicle.vehicle_track_id !== null) {
    parts.push(`Xe #${vehicle.vehicle_track_id}`)
  }
  if (vehicle.plate_track_id !== undefined && vehicle.plate_track_id !== null) {
    parts.push(`Biển số #${vehicle.plate_track_id}`)
  }
  return parts.join(' · ')
}

function frameScore(frame) {
  if (!frame) return 0
  if (Number.isFinite(Number(frame.combined_score))) return Number(frame.combined_score)
  const quality = Number(frame.quality_score) || 0
  const ocrConfidence = Math.max(Number(frame.ocr_confidence) || 0.1, 0.1)
  return quality * ocrConfidence
}

function ImageBox({ src, alt, fallback, dark = false }) {
  return (
    <div className={cx('flex h-24 items-center justify-center rounded-lg border border-[var(--color-border)]', dark ? 'bg-black' : 'bg-black/30')}>
      {src ? (
        <img src={src} alt={alt} className="max-h-full max-w-full object-contain" />
      ) : (
        <span className="text-xs text-[var(--color-text-subtle)]">{fallback}</span>
      )}
    </div>
  )
}

function imageSrc(value) {
  if (!value) return null
  if (value.startsWith('http') || value.startsWith('data:')) return value
  return `data:image/jpeg;base64,${value}`
}

function qualityColor(quality) {
  if (quality >= 0.8) return 'bg-emerald-300'
  if (quality >= 0.6) return 'bg-amber-300'
  return 'bg-red-300'
}
