import { useState } from 'react'
import IncidentDetail from './IncidentDetail'

function fmtTime(iso) {
  if (!iso) return '--:--'
  return new Date(iso).toLocaleTimeString()
}

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function IncidentCard({ incident }) {
  const [expanded, setExpanded] = useState(false)
  const { id, status, markedAt, windowStartSec, windowEndSec, vehicles, pct, error } = incident
  const vehArr = Object.values(vehicles || {})
  const primary = vehArr[0]

  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3 mb-2">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{id.slice(-6)}</span>
        <span>{fmtTime(markedAt)}</span>
      </div>
      <div className="text-xs text-slate-400 mt-1">
        Δ = {(windowEndSec - windowStartSec).toFixed(1)}s · {vehArr.length} xe
      </div>

      {status === 'pending' || status === 'processing' ? (
        <div className="mt-2 text-xs text-blue-300">
          Đang phân tích… {pct ? `${pct}%` : ''}
        </div>
      ) : status === 'failed' ? (
        <div className="mt-2 text-xs text-red-400">Lỗi: {error}</div>
      ) : (
        <>
          {primary && (
            <div className="mt-2 text-sm font-bold text-emerald-400">
              {displayPlateText(primary.plate)}
            </div>
          )}
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-xs text-slate-400 hover:text-white"
          >
            {expanded ? 'Ẩn' : 'Chi tiết'}
          </button>
          {expanded && <IncidentDetail incident={incident} />}
        </>
      )}
    </div>
  )
}
