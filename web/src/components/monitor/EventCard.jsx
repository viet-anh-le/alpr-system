import { useState } from 'react'

import { Badge, Progress } from '../ui'
import EventDetail from './EventDetail'

function fmtTime(iso) {
  if (!iso) return '--:--'
  return new Date(iso).toLocaleTimeString('vi')
}

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function EventCard({ event }) {
  const [expanded, setExpanded] = useState(false)
  const { id, status, markedAt, windowStartSec, windowEndSec, vehicles, pct, error } = event
  const vehicleList = Object.values(vehicles || {})
  const primary = vehicleList[0]
  const tone = status === 'completed' ? 'success' : status === 'failed' ? 'danger' : 'info'

  return (
    <article className="rounded-[var(--radius-panel)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="data-font text-xs text-[var(--color-text-subtle)]">#{id.slice(-8)} · {fmtTime(markedAt)}</p>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]">
            Window {(windowEndSec - windowStartSec).toFixed(1)}s · {vehicleList.length} vehicles
          </p>
        </div>
        <Badge tone={tone}>{status}</Badge>
      </div>

      {(status === 'pending' || status === 'processing') && (
        <div className="mt-3">
          <div className="mb-2 flex items-center justify-between text-xs text-[var(--color-text-muted)]">
            <span>Đang phân tích sự kiện</span>
            <span className="data-font">{pct ? `${pct}%` : 'queued'}</span>
          </div>
          <Progress value={pct || 8} />
        </div>
      )}

      {status === 'failed' && (
        <div className="mt-3 rounded-lg border border-red-300/30 bg-red-500/10 px-3 py-2 text-xs text-red-100">
          {error}
        </div>
      )}

      {status === 'completed' && (
        <>
          {primary && (
            <div className="mt-3 rounded-lg border border-emerald-300/20 bg-emerald-300/10 px-3 py-2">
              <p className="section-label text-emerald-100">Primary plate</p>
              <p className="plate-font mt-1 text-lg font-bold tracking-widest text-emerald-50">
                {displayPlateText(primary.plate)}
              </p>
            </div>
          )}
          <button
            type="button"
            onClick={() => setExpanded((value) => !value)}
            className="mt-3 text-sm font-semibold text-cyan-100 hover:text-cyan-50"
          >
            {expanded ? 'Ẩn evidence' : 'Xem evidence'}
          </button>
          {expanded && <EventDetail event={event} />}
        </>
      )}
    </article>
  )
}
