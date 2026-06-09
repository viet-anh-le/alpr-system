const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function IncidentDetail({ incident }) {
  const vehArr = Object.values(incident.vehicles || {})
  return (
    <div className="mt-2 space-y-2">
      {vehArr.map((v) => (
        <div key={v.track_id ?? v.id} className="bg-slate-900/50 rounded p-2">
          <div className="grid grid-cols-2 gap-px bg-slate-950 rounded overflow-hidden mb-2">
            <ImageBox
              src={imageSrc(v.vehicle_b64)}
              alt={`Xe ${v.track_id ?? v.id}`}
              fallback="Không có ảnh xe"
            />
            <ImageBox
              src={imageSrc(v.plate_b64)}
              alt={displayPlateText(v.plate)}
              fallback="Không có ảnh biển số"
            />
          </div>
          <div className="text-sm font-mono text-emerald-300">{displayPlateText(v.plate)}</div>
          <div className="text-[10px] text-slate-500">{v.cls} · {v.ocr_frames} frames</div>
          <TrackBuffer frames={v.track_buffer || []} />
        </div>
      ))}
    </div>
  )
}

function TrackBuffer({ frames }) {
  const sortedFrames = [...frames].sort(
    (a, b) => (b.quality_score ?? 0) - (a.quality_score ?? 0)
  )

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-slate-400 uppercase tracking-wider">
          Bộ đệm track
        </span>
        <span className="text-[10px] text-slate-500">{sortedFrames.length} ảnh</span>
      </div>
      {sortedFrames.length === 0 ? (
        <p className="text-[10px] text-slate-600 text-center py-3 bg-slate-950 rounded">
          Không có ảnh trong bộ đệm
        </p>
      ) : (
        <div className="grid grid-cols-3 gap-1.5">
          {sortedFrames.map((frame, index) => (
            <FrameCell key={`${frame.frame_index}-${frame.candidate_method || ''}-${index}`} frame={frame} />
          ))}
        </div>
      )}
    </div>
  )
}

function FrameCell({ frame }) {
  const quality = frame.quality_score ?? 0
  const src = imageSrc(frame.image_b64) || frame.image_url

  return (
    <div className="bg-slate-950 rounded overflow-hidden ring-1 ring-slate-800">
      <div className="h-14 bg-black flex items-center justify-center">
        {src ? (
          <img
            src={src}
            alt={`frame ${frame.frame_index}`}
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <span className="text-[9px] text-slate-700">no img</span>
        )}
      </div>
      <div className="h-1 bg-slate-800">
        <div
          className={`h-full ${qualityColor(quality)}`}
          style={{ width: `${Math.min(quality * 100, 100)}%` }}
        />
      </div>
      <div className="px-1 py-0.5 text-[9px] text-slate-500 text-center tabular-nums">
        f{frame.frame_index} · {quality.toFixed(2)}
      </div>
    </div>
  )
}

function ImageBox({ src, alt, fallback }) {
  return (
    <div className="h-16 bg-black flex items-center justify-center">
      {src ? (
        <img src={src} alt={alt} className="max-w-full max-h-full object-contain" />
      ) : (
        <span className="text-[10px] text-slate-700">{fallback}</span>
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
  if (quality >= 0.8) return 'bg-emerald-500'
  if (quality >= 0.6) return 'bg-amber-500'
  return 'bg-red-500'
}
