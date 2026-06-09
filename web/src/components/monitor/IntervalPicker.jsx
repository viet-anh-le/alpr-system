import { useState, useEffect } from 'react'

const MAX_INTERVAL = 30.0  // seconds

function fmt(t) {
  if (!Number.isFinite(t)) return '0:00'
  const m = Math.floor(t / 60), s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function IntervalPicker({
  duration, initialStart, initialEnd, onSeek, onAnalyze, onCancel,
}) {
  const [start, setStart] = useState(initialStart)
  const [end,   setEnd]   = useState(initialEnd)

  const delta = end - start
  const tooLong = delta > MAX_INTERVAL
  const valid = delta > 0 && !tooLong

  useEffect(() => { onSeek(start) }, [start])  // preview start
  useEffect(() => { onSeek(end) },   [end])    // preview end

  return (
    <div className="bg-slate-800/70 border border-slate-700 rounded-lg p-4 mt-3">
      <div className="text-xs text-slate-400 mb-2">Timeline</div>
      <div className="flex items-center gap-3 text-xs">
        <span>{fmt(0)}</span>
        <input
          type="range" min={0} max={duration} step={0.1}
          value={start}
          onChange={(e) => setStart(Math.min(parseFloat(e.target.value), end - 0.1))}
          className="flex-1"
        />
        <input
          type="range" min={0} max={duration} step={0.1}
          value={end}
          onChange={(e) => setEnd(Math.max(parseFloat(e.target.value), start + 0.1))}
          className="flex-1"
        />
        <span>{fmt(duration)}</span>
      </div>
      <div className="text-xs text-slate-400 mt-2">
        {fmt(start)} — {fmt(end)}  ·  Δ = {delta.toFixed(1)}s
        {tooLong && <span className="text-red-400 ml-2">(tối đa {MAX_INTERVAL}s)</span>}
      </div>
      <div className="flex gap-2 mt-3">
        <button
          onClick={() => onAnalyze(start, end)}
          disabled={!valid}
          className={`text-xs px-4 py-2 rounded font-medium ${
            valid ? 'bg-blue-600 hover:bg-blue-500 text-white'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
          }`}
        >
          Phân tích
        </button>
        <button
          onClick={onCancel}
          className="text-xs px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-slate-300"
        >
          Hủy
        </button>
      </div>
    </div>
  )
}
