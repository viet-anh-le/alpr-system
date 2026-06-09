export default function ProgressBar({ frame, total, pct, status }) {
  const isDone  = status === 'done'
  const isError = status === 'error'

  const label =
    isDone             ? 'Hoàn tất — phát hiện xe thành công'
    : isError          ? 'Đã xảy ra lỗi'
    : status === 'uploading' ? 'Đang tải lên…'
    : 'Đang xử lý video…'

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {isDone ? (
            <span className="text-emerald-500 text-lg">✅</span>
          ) : isError ? (
            <span className="text-red-500 text-lg">❌</span>
          ) : (
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          )}
          <span className="text-sm font-medium text-slate-700">{label}</span>
        </div>
        <span className="text-sm font-bold text-blue-600 tabular-nums">{pct}%</span>
      </div>

      <div className="bg-slate-100 rounded-full h-2.5 overflow-hidden">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-300 ease-linear"
          style={{ width: `${pct}%` }}
        />
      </div>

      {frame > 0 && (
        <p className="text-xs text-slate-400 mt-2">
          Frame {frame.toLocaleString('vi')} / {total.toLocaleString('vi')}
        </p>
      )}
    </div>
  )
}
