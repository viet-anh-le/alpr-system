/**
 * LiveFrame — left panel of the 2-column layout.
 *
 * During processing : shows the latest annotated frame streamed from the backend
 *                     (bounding boxes drawn by OpenCV).
 * After done        : switches to the original video so the user can review.
 */
export default function LiveFrame({ frameB64, videoUrl, progress, status }) {
  const isDone  = status === 'done'
  const isError = status === 'error'
  const hasFrame = !!frameB64

  return (
    <div className="relative bg-black rounded-2xl overflow-hidden shadow-xl select-none">

      {/* ── Annotated frame (during processing) ── */}
      {hasFrame && !isDone && (
        <img
          src={`data:image/jpeg;base64,${frameB64}`}
          alt="Annotated frame"
          className="w-full block"
          style={{ maxHeight: '70vh', objectFit: 'contain' }}
          draggable={false}
        />
      )}

      {/* ── Original video (when done, or while uploading before first frame) ── */}
      {(isDone || !hasFrame) && videoUrl && (
        <video
          key={videoUrl}
          src={videoUrl}
          controls
          muted
          autoPlay={!isDone}
          loop={isDone}
          className="w-full block"
          style={{ maxHeight: '70vh', objectFit: 'contain' }}
        />
      )}

      {/* ── Placeholder when no video yet ── */}
      {!videoUrl && !hasFrame && (
        <div
          className="w-full flex items-center justify-center text-slate-600"
          style={{ height: '40vh' }}
        >
          <svg className="w-12 h-12 opacity-30" fill="none" viewBox="0 0 24 24"
               stroke="currentColor" strokeWidth={1}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53
                     l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0
                     0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
          </svg>
        </div>
      )}

      {/* ── Progress overlay (always shown while not idle) ── */}
      {videoUrl && (
        <div className="absolute bottom-0 left-0 right-0
                        bg-gradient-to-t from-black/85 to-transparent
                        px-4 pb-3 pt-10 pointer-events-none">
          <div className="flex items-center justify-between text-xs text-white/80 mb-1.5">
            <div className="flex items-center gap-1.5">
              {!isDone && !isError && (
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              )}
              <span className="font-medium drop-shadow">
                {isDone             ? '✅ Phân tích hoàn tất'
                 : isError          ? '❌ Đã xảy ra lỗi'
                 : status === 'uploading' ? '⬆ Đang tải lên…'
                 : '⟳ Đang phân tích…'}
              </span>
            </div>
            <span className="font-bold tabular-nums drop-shadow">{progress.pct}%</span>
          </div>

          <div className="bg-white/20 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-300 ease-linear
                          ${isDone ? 'bg-emerald-400' : 'bg-blue-400'}`}
              style={{ width: `${progress.pct}%` }}
            />
          </div>

          {progress.frame > 0 && !isDone && (
            <p className="text-[10px] text-white/40 mt-1 tabular-nums">
              Frame {progress.frame.toLocaleString('vi')} / {progress.total.toLocaleString('vi')}
            </p>
          )}
        </div>
      )}

      {/* ── Legend overlay (top-right, shown during processing) ── */}
      {hasFrame && !isDone && (
        <div className="absolute top-3 right-3 bg-black/60 rounded-lg px-2.5 py-1.5
                        text-[10px] space-y-0.5 pointer-events-none">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 rounded-full inline-block bg-yellow-400" style={{height:3}} />
            <span className="text-white/70">Đang OCR</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 inline-block bg-green-500 rounded-full" style={{height:3}} />
            <span className="text-white/70">Đã xác nhận</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 inline-block bg-white/40 rounded-full" style={{height:3}} />
            <span className="text-white/70">Đang theo dõi</span>
          </div>
        </div>
      )}
    </div>
  )
}
