import { useEffect, useRef, useState } from 'react'
import IntervalPicker from './IntervalPicker'

export default function UploadViewer({ videoUrl, onMark }) {
  const videoRef = useRef(null)
  const [duration, setDuration] = useState(0)
  const [picking, setPicking]   = useState(false)
  const [initialRange, setInitialRange] = useState([0, 1])

  const handleStartPicking = () => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    const t = v.currentTime
    const start = Math.max(0, t - 10)
    const end   = Math.min(duration, t + 5)
    setInitialRange([start, end])
    setPicking(true)
  }

  const handleSeek = (t) => {
    if (videoRef.current) videoRef.current.currentTime = t
  }

  return (
    <div className="bg-black rounded-lg overflow-hidden">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="w-full aspect-video object-contain"
        onLoadedMetadata={(e) => setDuration(e.target.duration)}
      />
      {!picking ? (
        <div className="bg-slate-800/70 border-t border-slate-700 p-3">
          <button
            onClick={handleStartPicking}
            className="text-xs px-4 py-2 rounded bg-red-600 hover:bg-red-500 text-white font-medium"
          >
            🚩 Mark Interval
          </button>
        </div>
      ) : (
        <IntervalPicker
          duration={duration}
          initialStart={initialRange[0]}
          initialEnd={initialRange[1]}
          onSeek={handleSeek}
          onAnalyze={(start, end) => { setPicking(false); onMark(start, end) }}
          onCancel={() => setPicking(false)}
        />
      )}
    </div>
  )
}
