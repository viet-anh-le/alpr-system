import { useState } from 'react'
import PlateDisplay from './PlateDisplay'
import ImageModal from './ImageModal'
import TrackBufferModal from './TrackBufferModal'

const CLS_LABEL = { car: 'Ô tô', motorcycle: 'Xe máy', motorbike_rider: 'Xe máy', bus: 'Xe buýt', truck: 'Xe tải' }
const CLS_ICON = { car: '🚗', motorcycle: '🏍️', motorbike_rider: '🏍️', bus: '🚌', truck: '🚛' }
const CLUSTER_COLORS = [
  { bg: 'bg-blue-900/40', border: 'border-blue-600', badge: 'bg-blue-600', label: 'Cụm' },
  { bg: 'bg-amber-900/40', border: 'border-amber-600', badge: 'bg-amber-600', label: 'Cụm' },
  { bg: 'bg-emerald-900/40', border: 'border-emerald-600', badge: 'bg-emerald-600', label: 'Cụm' },
]

export default function VehicleCard({ vehicle, jobId }) {
  const { id, cls, plate, chars, done, plate_b64, vehicle_b64, ocr_frames, clusters } = vehicle
  const [copied, setCopied] = useState(false)
  const [zoomedImg, setZoomedImg] = useState(null)
  const [showBuffer, setShowBuffer] = useState(false)

  const cleanPlate = (plate || '').replace(/#/g, '')
  const hasUnknown = (plate || '').includes('#')
  const frames = ocr_frames || 0
  const avgConf = chars?.length
    ? Math.round(chars.reduce((s, [, p]) => s + p, 0) / chars.length * 100)
    : 0
  const hasClusters = clusters && clusters.length > 1

  const copy = () => {
    navigator.clipboard.writeText(cleanPlate).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="border-b border-slate-700 last:border-0 bg-slate-800">
      {/* ── Header row ── */}
      <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5">
        <div className="flex items-center gap-1.5">
          <span className="text-sm leading-none">{CLS_ICON[cls] || '🚘'}</span>
          <span className="text-[11px] font-medium text-slate-300">
            {CLS_LABEL[cls] || cls || 'Xe'}
          </span>
          <span className="text-[10px] text-slate-500">#{id}</span>
        </div>

        {done ? (
          <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-400">
            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 45 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            Đã xác nhận
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-[10px] text-amber-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Đang phân tích
          </span>
        )}
      </div>

      {/* ── Two images side-by-side ── */}
      <div className="grid grid-cols-2 gap-px bg-slate-900 mx-3 rounded-lg overflow-hidden mb-2">
        {/* Left: full vehicle image */}
        <div
          className={`bg-slate-950 flex items-center justify-center relative group ${vehicle_b64 ? 'cursor-zoom-in' : ''}`}
          style={{ height: 90 }}
          onClick={() => vehicle_b64 && setZoomedImg({ src: `data:image/jpeg;base64,${vehicle_b64}`, alt: `Phương tiện #${id} (${CLS_LABEL[cls] || cls})` })}
        >
          {vehicle_b64 ? (
            <>
              <img
                src={`data:image/jpeg;base64,${vehicle_b64}`}
                alt="Phương tiện"
                className="max-w-full max-h-full object-contain transition-transform duration-300 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors pointer-events-none" />
            </>
          ) : (
            <VehiclePlaceholder />
          )}
        </div>

        {/* Right: plate crop image */}
        <div
          className={`bg-black flex items-center justify-center relative group ${plate_b64 ? 'cursor-zoom-in' : ''}`}
          style={{ height: 90 }}
          onClick={() => plate_b64 && setZoomedImg({ src: `data:image/jpeg;base64,${plate_b64}`, alt: `Biển số #${id} (${cleanPlate || 'Đang nhận dạng'})` })}
        >
          {plate_b64 ? (
            <>
              <img
                src={`data:image/jpeg;base64,${plate_b64}`}
                alt="Biển số"
                className="max-w-full max-h-full object-contain transition-transform duration-300 group-hover:scale-105"
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors pointer-events-none" />
            </>
          ) : (
            <PlatePlaceholder />
          )}
        </div>
      </div>

      {/* ── OCR result ── */}
      <div className="px-3 pb-2.5">
        {/* Char confidence blocks */}
        <div className="mb-1.5">
          <PlateDisplay chars={chars} compact />
        </div>

        {/* Large plate text */}
        {cleanPlate ? (
          <p className="plate-font text-white font-bold text-base tracking-widest leading-none">
            {cleanPlate}
          </p>
        ) : (
          <p className="text-slate-500 text-xs italic">Đang nhận dạng…</p>
        )}

        {/* Stats + copy */}
        <div className="flex items-center justify-between mt-1.5">
          <span className="text-[10px] text-slate-500 tabular-nums">
            {frames < 2
              ? <span className="text-amber-500">⚠ {frames} frame</span>
              : `${frames} frame`
            }
            {avgConf > 0 && (
              <span className="ml-1">
                · <span className={avgConf >= 90 ? 'text-emerald-400' : avgConf >= 70 ? 'text-amber-400' : 'text-red-400'}>
                  {avgConf}%
                </span>
              </span>
            )}
            {hasClusters && (
              <span className="ml-1 text-sky-400">
                · {clusters.length} cụm
              </span>
            )}
          </span>

          <div className="flex items-center gap-1.5">
            {done && jobId && (
              <button
                onClick={() => setShowBuffer(true)}
                className="text-[10px] px-2 py-0.5 rounded-md transition-colors bg-slate-700 hover:bg-blue-700 text-white border border-slate-600 hover:border-blue-500"
              >
                Bộ đệm
              </button>
            )}
            {!hasUnknown && cleanPlate && (
              <button
                onClick={copy}
                className={`text-[10px] px-2 py-0.5 rounded-md transition-colors ${
                  copied
                    ? 'bg-emerald-900/50 text-emerald-400 border border-emerald-700'
                    : 'bg-slate-700 hover:bg-slate-600 text-slate-300 border border-slate-600'
                }`}
              >
                {copied ? '✓ Đã sao chép' : 'Sao chép'}
              </button>
            )}
          </div>
        </div>

        {/* ── Multi-cluster display ── */}
        {hasClusters && (
          <div className="mt-2 space-y-2">
            {clusters.map((cluster, idx) => {
              const color = CLUSTER_COLORS[idx % CLUSTER_COLORS.length]
              const clusterClean = (cluster.plate || '').replace(/#/g, '')
              const isPrimary = idx === 0
              return (
                <div key={idx} className={`rounded-md border ${color.border} ${color.bg} p-2`}>
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${color.badge} text-white`}>
                      {color.label} {idx + 1}
                      {isPrimary ? ' (chính)' : ''}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {cluster.frame_count} frame · {Math.round((cluster.confidence || 0) * 100)}%
                    </span>
                  </div>

                  {/* Plate text */}
                  {clusterClean ? (
                    <p className={`plate-font font-bold text-sm tracking-widest leading-none mb-1.5 ${isPrimary ? 'text-white' : 'text-slate-300'}`}>
                      {clusterClean}
                    </p>
                  ) : (
                    <p className="text-slate-500 text-[10px] italic mb-1.5">Không đọc được</p>
                  )}

                  {/* Per-char confidence */}
                  {cluster.chars && cluster.chars.length > 0 && (
                    <div className="flex flex-wrap gap-0.5 mb-1.5">
                      {cluster.chars.map(([ch, conf], ci) => {
                        const tone = conf >= 0.9 ? 'bg-emerald-600' : conf >= 0.7 ? 'bg-amber-600' : 'bg-red-600'
                        return (
                          <span key={ci} className={`inline-flex items-center justify-center h-5 w-4 rounded text-[9px] font-bold text-white ${tone}`} title={`${ch}: ${Math.round(conf * 100)}%`}>
                            {ch === '#' ? '?' : ch}
                          </span>
                        )
                      })}
                    </div>
                  )}

                  {/* Plate crop image */}
                  {cluster.plate_b64 ? (
                    <div
                      className="bg-black rounded overflow-hidden cursor-zoom-in"
                      style={{ height: 70 }}
                      onClick={() => setZoomedImg({
                        src: `data:image/jpeg;base64,${cluster.plate_b64}`,
                        alt: `Biển số cụm ${idx + 1}: ${clusterClean || '?'}`,
                      })}
                    >
                      <img
                        src={`data:image/jpeg;base64,${cluster.plate_b64}`}
                        alt={`Cụm ${idx + 1}`}
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  ) : (
                    <div className="bg-slate-900 rounded flex items-center justify-center" style={{ height: 70 }}>
                      <span className="text-[9px] text-slate-700">Không có ảnh</span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Image Modal for Zoom */}
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

function VehiclePlaceholder() {
  return (
    <div className="flex flex-col items-center gap-1 text-slate-700">
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h1.125c.621 0 1.129-.504 1.09-1.124a17.902 17.902 0 00-3.213-9.193 2.056 2.056 0 00-1.58-.86H14.25M16.5 18.75h-2.25m0-11.177v-.958c0-.568-.422-1.048-.987-1.106a48.554 48.554 0 00-10.026 0 1.106 1.106 0 00-.987 1.106v7.635m12-6.677v6.677m0 4.5v-4.5m0 0h-12" />
      </svg>
      <span className="text-[9px]">Chờ ảnh xe</span>
    </div>
  )
}

function PlatePlaceholder() {
  return (
    <div className="flex flex-col items-center gap-1 text-slate-700">
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.2}>
        <rect x="3" y="7" width="18" height="10" rx="1.5" stroke="currentColor" fill="none" />
        <path d="M7 10h10M7 14h5" stroke="currentColor" strokeLinecap="round" />
      </svg>
      <span className="text-[9px]">Chờ biển số</span>
    </div>
  )
}
