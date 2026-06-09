import { useEffect, useState } from 'react'

const qualityColor = (q) =>
  q >= 0.8 ? 'bg-emerald-500' : q >= 0.6 ? 'bg-amber-500' : 'bg-red-500'

const qualityText = (q) =>
  q >= 0.8 ? 'text-emerald-400' : q >= 0.6 ? 'text-amber-400' : 'text-red-400'

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function TrackBufferModal({ vehicle, jobId, onClose }) {
  const [record, setRecord]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setRecord(null)

    fetch(`/records/${jobId}/${vehicle.id}`)
      .then(r => {
        if (!r.ok) throw new Error(r.status === 404 ? 'Dữ liệu chưa được lưu xong, thử lại sau.' : `Lỗi HTTP ${r.status}`)
        return r.json()
      })
      .then(data => { setRecord(data); setLoading(false) })
      .catch(err  => { setError(err.message); setLoading(false) })
  }, [jobId, vehicle.id])

  const frames       = record?.track_buffer ?? []
  const sortedFrames = [...frames].sort((a, b) => b.quality_score - a.quality_score)
  const bestIdx      = record?.best_plate_frame?.frame_index
  const votes        = Object.entries(record?.ocr_vote_summary ?? {}).sort((a, b) => b[1] - a[1])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
      onClick={e => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl flex flex-col overflow-hidden"
           style={{ maxHeight: '88vh' }}>

        {/* ── Header ── */}
        <div className="bg-blue-700 px-4 py-3 flex items-center justify-between flex-shrink-0">
          <div>
            <p className="text-white text-sm font-bold leading-tight">
              Track #{vehicle.id}
              {record?.plate_text && (
                <span className="ml-2 font-mono tracking-widest">{displayPlateText(record.plate_text)}</span>
              )}
            </p>
            <p className="text-blue-200 text-[11px] mt-0.5">
              {loading ? 'Đang tải…'
                : record ? `${frames.length} ảnh · ${{ segment_vote: 'Segment-vote OCR', prob_vote: 'Prob-vote OCR', paddle_segment_vote: 'Paddle segment-vote OCR', paddle_prob_vote: 'Paddle prob-vote OCR' }[record.ocr_method] ?? record.ocr_method} · conf ${Math.round((record.plate_text_confidence ?? 0) * 100)}%`
                : ''}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-blue-200 hover:text-white w-8 h-8 flex items-center justify-center
                       rounded-full hover:bg-blue-600 transition-colors text-lg leading-none"
          >
            ✕
          </button>
        </div>

        {/* ── Body ── */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* Loading skeleton */}
          {loading && <Skeleton />}

          {/* Error */}
          {error && !loading && (
            <div className="flex flex-col items-center justify-center py-10 gap-2 text-center">
              <span className="text-3xl">⚠️</span>
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {record && (
            <>
              {/* Best images */}
              <div className="flex gap-3">
                <ImageBox
                  url={record.vehicle_thumbnail_url}
                  label="Phương tiện"
                  height={96}
                />
                <ImageBox
                  url={record.best_plate_frame?.image_url}
                  label={`Biển số tốt nhất · q = ${(record.best_plate_frame?.quality_score ?? 0).toFixed(2)}`}
                  height={96}
                  highlight
                />
              </div>

              {/* OCR vote summary */}
              {votes.length > 0 && (
                <div className="bg-slate-900/60 rounded-xl px-3 py-2.5">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-2">
                    Kết quả bỏ phiếu OCR
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {votes.map(([text, count]) => (
                      <span
                        key={text}
                        className={`text-xs px-2.5 py-1 rounded-lg font-mono font-bold
                          ${displayPlateText(text) === displayPlateText(record.plate_text)
                            ? 'bg-emerald-600 text-white ring-1 ring-emerald-400'
                            : 'bg-slate-700 text-slate-300'}`}
                      >
                        {displayPlateText(text)}
                        <span className="ml-1.5 opacity-60 font-normal">×{count}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Track buffer grid */}
              <div>
                <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-2">
                  Bộ đệm track — {frames.length} ảnh (sắp xếp theo chất lượng)
                </p>
                {sortedFrames.length === 0 ? (
                  <p className="text-slate-500 text-sm text-center py-6">Không có ảnh trong bộ đệm</p>
                ) : (
                  <div className="grid grid-cols-4 gap-2">
                    {sortedFrames.map((frame) => (
                      <FrameCell
                        key={frame.frame_index}
                        frame={frame}
                        isBest={frame.frame_index === bestIdx}
                      />
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Sub-components ─────────────────────────────────────────────────────── */

function ImageBox({ url, label, height, highlight }) {
  return (
    <div className={`flex-1 rounded-xl overflow-hidden flex flex-col
      ${highlight ? 'ring-2 ring-emerald-500' : 'ring-1 ring-slate-700'}`}
    >
      <div className="flex-1 bg-slate-950 flex items-center justify-center" style={{ height }}>
        {url
          ? <img src={url} alt={label} className="max-w-full max-h-full object-contain" />
          : <span className="text-slate-600 text-[11px]">Không có ảnh</span>
        }
      </div>
      <p className="text-center text-[10px] text-slate-400 bg-slate-900 py-1 px-2 truncate">
        {label}
      </p>
    </div>
  )
}

function FrameCell({ frame, isBest }) {
  const q = frame.quality_score ?? 0

  return (
    <div className={`rounded-lg overflow-hidden bg-slate-900 flex flex-col
      ${isBest ? 'ring-2 ring-emerald-500' : 'ring-1 ring-slate-700'}`}
    >
      {/* Image */}
      <div className="relative bg-black flex items-center justify-center" style={{ height: 64 }}>
        {frame.image_url
          ? <img src={frame.image_url} alt={`frame ${frame.frame_index}`}
                 className="max-w-full max-h-full object-contain" />
          : <span className="text-slate-700 text-[9px]">no img</span>
        }
        {isBest && (
          <span className="absolute top-0.5 right-0.5 bg-emerald-600 text-white
                           text-[8px] font-bold px-1 py-px rounded">
            BEST
          </span>
        )}
      </div>

      {/* Quality bar */}
      <div className="h-1 bg-slate-700">
        <div
          className={`h-full ${qualityColor(q)}`}
          style={{ width: `${Math.min(q * 100, 100)}%` }}
        />
      </div>

      {/* Label */}
      <p className={`text-center text-[9px] py-1 ${qualityText(q)}`}>
        f{frame.frame_index} · {q.toFixed(2)}
      </p>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="animate-pulse space-y-3">
      <div className="flex gap-3">
        <div className="flex-1 h-24 bg-slate-700 rounded-xl" />
        <div className="flex-1 h-24 bg-slate-700 rounded-xl" />
      </div>
      <div className="h-12 bg-slate-700 rounded-xl" />
      <div className="grid grid-cols-4 gap-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-20 bg-slate-700 rounded-lg" />
        ))}
      </div>
    </div>
  )
}
