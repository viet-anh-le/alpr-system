import { useRef, useState } from 'react'

import { Button } from '../ui'
import IntervalPicker from './IntervalPicker'

export default function UploadViewer({ videoUrl, onMark }) {
  const videoRef = useRef(null)
  const [duration, setDuration] = useState(0)
  const [picking, setPicking] = useState(false)
  const [initialRange, setInitialRange] = useState([0, 1])

  const handleStartPicking = () => {
    const video = videoRef.current
    if (!video) return
    video.pause()
    const current = video.currentTime
    setInitialRange([Math.max(0, current - 10), Math.min(duration, current + 5)])
    setPicking(true)
  }

  const handleSeek = (time) => {
    if (videoRef.current) videoRef.current.currentTime = time
  }

  return (
    <section className="surface-panel overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Quan sát từ video tải lên</p>
          <h2 className="mt-1 text-lg font-bold">Chọn đoạn để phân tích</h2>
        </div>
        {!picking && <Button size="sm" variant="primary" onClick={handleStartPicking}>Đánh dấu đoạn</Button>}
      </div>
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="aspect-video w-full bg-black object-contain"
        onLoadedMetadata={(event) => setDuration(event.target.duration)}
      />
      {picking && (
        <IntervalPicker
          duration={duration}
          initialStart={initialRange[0]}
          initialEnd={initialRange[1]}
          onSeek={handleSeek}
          onAnalyze={(start, end) => {
            setPicking(false)
            onMark(start, end)
          }}
          onCancel={() => setPicking(false)}
        />
      )}
    </section>
  )
}
