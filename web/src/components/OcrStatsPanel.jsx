import { useState } from 'react'
import PlateDisplay from './PlateDisplay'
import ImageModal from './ImageModal'
import TrackBufferModal from './TrackBufferModal'

const CLS_LABEL = { car: 'Ô tô', motorcycle: 'Xe máy', bus: 'Xe buýt', truck: 'Xe tải' }
const CLS_ICON  = { car: '🚗',   motorcycle: '🏍️',    bus: '🚌',      truck: '🚛'      }

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function OcrStatsPanel({ vehicles, rejectedVehicles, jobId }) {
  const totalOcr     = vehicles.length + rejectedVehicles.length
  const validCount   = vehicles.length
  const invalidCount = rejectedVehicles.length

  if (totalOcr === 0) return null

  return (
    <div className="bg-slate-800 rounded-2xl overflow-hidden shadow-lg mt-4">
      {/* ── Header ── */}
      <div className="bg-indigo-700 px-4 py-2.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24"
               stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
          </svg>
          <span className="text-white text-xs font-bold uppercase tracking-widest">
            Thống kê OCR
          </span>
        </div>
        <span className="bg-indigo-600 text-white text-xs font-bold px-2 py-0.5 rounded-full">
          {totalOcr}
        </span>
      </div>

      {/* ── Summary counters ── */}
      <div className="grid grid-cols-3 gap-px bg-slate-900">
        <StatBox
          label="Đã OCR"
          value={totalOcr}
          icon="📋"
          color="text-blue-400"
          bg="bg-blue-500/10"
        />
        <StatBox
          label="Hợp lệ"
          value={validCount}
          icon="✅"
          color="text-emerald-400"
          bg="bg-emerald-500/10"
        />
        <StatBox
          label="Không hợp lệ"
          value={invalidCount}
          icon="❌"
          color="text-red-400"
          bg="bg-red-500/10"
        />
      </div>

      {/* ── Accuracy bar ── */}
      {totalOcr > 0 && (
        <div className="px-4 py-2 bg-slate-800/80">
          <div className="flex items-center justify-between text-[10px] mb-1">
            <span className="text-slate-400">Tỷ lệ hợp lệ</span>
            <span className="text-white font-bold tabular-nums">
              {Math.round(validCount / totalOcr * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-emerald-500 to-emerald-400"
              style={{ width: `${validCount / totalOcr * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* ── Invalid plates list ── */}
      {invalidCount > 0 && (
        <div className="border-t border-slate-700">
          <div className="px-4 py-2 bg-slate-900/60">
            <p className="text-[10px] text-red-400 uppercase tracking-wider font-semibold flex items-center gap-1.5">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round"
                      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              Biển số không hợp lệ ({invalidCount})
            </p>
          </div>
          <div className="max-h-[300px] overflow-y-auto">
            {rejectedVehicles.map(rv => (
              <RejectedCard key={rv.id} vehicle={rv} jobId={jobId} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Stat counter box ── */
function StatBox({ label, value, icon, color, bg }) {
  return (
    <div className={`${bg} bg-slate-800 px-3 py-3 flex flex-col items-center gap-1`}>
      <span className="text-lg leading-none">{icon}</span>
      <span className={`text-lg font-bold tabular-nums ${color}`}>{value}</span>
      <span className="text-[10px] text-slate-400">{label}</span>
    </div>
  )
}

/* ── Rejected plate card ── */
function RejectedCard({ vehicle, jobId }) {
  const { id, cls, plate, chars, plate_b64, vehicle_b64, ocr_frames, vote_summary } = vehicle
  const [expanded, setExpanded]     = useState(false)
  const [zoomedImg, setZoomedImg]   = useState(null)
  const [showBuffer, setShowBuffer] = useState(false)
  const plateText = displayPlateText(plate)

  const avgConf = chars?.length
    ? Math.round(chars.reduce((s, [, p]) => s + p, 0) / chars.length * 100)
    : 0

  const votes = vote_summary ? Object.entries(vote_summary).sort((a, b) => b[1] - a[1]) : []

  return (
    <div className="border-b border-slate-700/50 last:border-0">
      {/* Collapsed row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2.5 px-4 py-2 hover:bg-slate-700/30 transition-colors text-left"
      >
        <span className="text-sm leading-none">{CLS_ICON[cls] || '🚘'}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-medium text-slate-300">
              {CLS_LABEL[cls] || cls || 'Xe'}
            </span>
            <span className="text-[10px] text-slate-500">#{id}</span>
            <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-red-900/40 text-red-400 font-medium">
              Không hợp lệ
            </span>
          </div>
          <p className="plate-font text-white/70 text-xs tracking-wider mt-0.5 truncate">
            {plateText || '—'}
          </p>
        </div>
        <svg
          className={`w-3.5 h-3.5 text-slate-500 flex-shrink-0 transition-transform duration-200
                      ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="px-4 pb-3 space-y-2.5 animate-fade-in">
          {/* Images */}
          <div className="grid grid-cols-2 gap-px bg-slate-900 rounded-lg overflow-hidden">
            <div
              className={`bg-slate-950 flex items-center justify-center ${vehicle_b64 ? 'cursor-zoom-in' : ''}`}
              style={{ height: 80 }}
              onClick={() => vehicle_b64 && setZoomedImg({
                src: `data:image/jpeg;base64,${vehicle_b64}`,
                alt: `Phương tiện #${id}`
              })}
            >
              {vehicle_b64
                ? <img src={`data:image/jpeg;base64,${vehicle_b64}`} alt="Xe"
                       className="max-w-full max-h-full object-contain" />
                : <span className="text-slate-700 text-[10px]">Không có ảnh</span>
              }
            </div>
            <div
              className={`bg-black flex items-center justify-center ${plate_b64 ? 'cursor-zoom-in' : ''}`}
              style={{ height: 80 }}
              onClick={() => plate_b64 && setZoomedImg({
                src: `data:image/jpeg;base64,${plate_b64}`,
                alt: `Biển số #${id}`
              })}
            >
              {plate_b64
                ? <img src={`data:image/jpeg;base64,${plate_b64}`} alt="Biển số"
                       className="max-w-full max-h-full object-contain" />
                : <span className="text-slate-700 text-[10px]">Không có ảnh</span>
              }
            </div>
          </div>

          {/* Char confidence blocks */}
          <div>
            <p className="text-[10px] text-slate-400 mb-1">Kết quả OCR:</p>
            <PlateDisplay chars={chars} />
          </div>

          {/* Stats row */}
          <div className="flex items-center gap-3 text-[10px] text-slate-500">
            <span>{ocr_frames || 0} frame</span>
            {avgConf > 0 && (
              <span>
                Conf:&nbsp;
                <span className={avgConf >= 90 ? 'text-emerald-400' : avgConf >= 70 ? 'text-amber-400' : 'text-red-400'}>
                  {avgConf}%
                </span>
              </span>
            )}
          </div>

          {/* Vote summary */}
          {votes.length > 0 && (
            <div className="bg-slate-900/60 rounded-lg px-2.5 py-2">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider mb-1.5">
                Kết quả bỏ phiếu OCR
              </p>
              <div className="flex flex-wrap gap-1.5">
                {votes.map(([text, count]) => (
                  <span
                    key={text}
                    className="text-[10px] px-2 py-0.5 rounded-md font-mono font-bold bg-slate-700 text-slate-300"
                  >
                    {displayPlateText(text)}
                    <span className="ml-1 opacity-60 font-normal">×{count}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Track buffer button */}
          {jobId && (
            <button
              onClick={() => setShowBuffer(true)}
              className="text-[10px] px-2.5 py-1 rounded-md transition-colors
                         bg-slate-700 hover:bg-blue-700 text-slate-300 hover:text-white
                         border border-slate-600 hover:border-blue-500"
            >
              Xem bộ đệm track
            </button>
          )}
        </div>
      )}

      {/* Image zoom modal */}
      {zoomedImg && (
        <ImageModal
          src={zoomedImg.src}
          alt={zoomedImg.alt}
          onClose={() => setZoomedImg(null)}
        />
      )}

      {/* Track Buffer Modal */}
      {showBuffer && jobId && (
        <TrackBufferModal
          vehicle={vehicle}
          jobId={jobId}
          onClose={() => setShowBuffer(false)}
        />
      )}
    </div>
  )
}
