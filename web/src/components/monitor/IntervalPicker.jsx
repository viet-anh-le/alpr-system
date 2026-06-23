import { useEffect, useState } from 'react'

import { Button, Progress } from '../ui'

const MAX_INTERVAL = 30.0

function fmt(time) {
  if (!Number.isFinite(time)) return '0:00'
  const minutes = Math.floor(time / 60)
  const seconds = Math.floor(time % 60)
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

export default function IntervalPicker({ duration, initialStart, initialEnd, onSeek, onAnalyze, onCancel }) {
  const [start, setStart] = useState(initialStart)
  const [end, setEnd] = useState(initialEnd)
  const delta = end - start
  const tooLong = delta > MAX_INTERVAL
  const valid = delta > 0 && !tooLong

  useEffect(() => { onSeek(start) }, [start]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { onSeek(end) }, [end]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="section-label">Incident window</p>
          <p className="mt-1 data-font text-sm text-[var(--color-text-muted)]">
            {fmt(start)} → {fmt(end)} · Δ {delta.toFixed(1)}s
          </p>
        </div>
        <span className={tooLong ? 'text-sm font-semibold text-red-100' : 'text-sm text-[var(--color-text-muted)]'}>
          tối đa {MAX_INTERVAL}s
        </span>
      </div>
      <Progress value={duration ? (end / duration) * 100 : 0} tone={tooLong ? 'danger' : 'info'} className="mb-4" />
      <div className="grid gap-3 md:grid-cols-2">
        <label className="space-y-1 text-xs text-[var(--color-text-muted)]">
          Start {fmt(start)}
          <input
            type="range"
            min={0}
            max={duration}
            step={0.1}
            value={start}
            onChange={(event) => setStart(Math.min(parseFloat(event.target.value), end - 0.1))}
            className="w-full"
          />
        </label>
        <label className="space-y-1 text-xs text-[var(--color-text-muted)]">
          End {fmt(end)}
          <input
            type="range"
            min={0}
            max={duration}
            step={0.1}
            value={end}
            onChange={(event) => setEnd(Math.max(parseFloat(event.target.value), start + 0.1))}
            className="w-full"
          />
        </label>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="primary" disabled={!valid} onClick={() => onAnalyze(start, end)}>
          Phân tích interval
        </Button>
        <Button variant="ghost" onClick={onCancel}>Hủy</Button>
      </div>
    </div>
  )
}
