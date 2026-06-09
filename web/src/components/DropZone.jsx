import { useRef, useState } from 'react'

const PREPROCESS_OPTIONS = [
  { value: 'none', label: 'Không xử lý' },
  { value: 'night', label: 'Ban đêm' },
  { value: 'low_contrast', label: 'Tương phản thấp' },
  { value: 'fog', label: 'Sương mù' },
  { value: 'rain', label: 'Mưa / nhiễu' },
  { value: 'glare', label: 'Chói sáng' },
]

const OCR_OPTIONS = [
  { value: 'default', label: 'Mặc định' },
  { value: 'smalllpr_ctc', label: 'SmallLPR CTC' },
  { value: 'parseq', label: 'PARSeq' },
  { value: 'yolov5_char', label: 'YOLOv5 Char' },
]

function formatBytes(b) {
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB'
  return (b / 1048576).toFixed(1) + ' MB'
}

export default function DropZone({
  onFileSelect,
  dark = false,
  preprocessMode = 'none',
  onPreprocessModeChange,
  ocrBackend = 'default',
  onOcrBackendChange,
}) {
  const inputRef           = useRef(null)
  const [file, setFile]    = useState(null)
  const [drag, setDrag]    = useState(false)
  const [loading, setLoad] = useState(false)

  const pick = (f) => { if (f) setFile(f) }

  const start = async () => {
    if (!file || loading) return
    setLoad(true)
    await onFileSelect(file, preprocessMode, ocrBackend)
    setLoad(false)
  }

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e)  => { e.preventDefault(); setDrag(true)  }}
        onDragLeave={()  => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}
        className={`border-2 border-dashed rounded-2xl flex flex-col items-center
                    justify-center gap-3 py-14 px-6 cursor-pointer transition-colors
                    ${dark
                      ? drag
                        ? 'border-blue-500 bg-blue-900/20'
                        : 'border-slate-600 bg-slate-800 hover:border-blue-500 hover:bg-blue-900/20'
                      : drag
                        ? 'border-blue-400 bg-blue-50/30'
                        : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/30'
                    }`}
      >
        <div className="w-16 h-16 bg-blue-50 rounded-2xl flex items-center justify-center">
          <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24"
               stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5
                     m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>
        <div className="text-center">
          <p className={`font-semibold ${dark ? 'text-slate-200' : 'text-slate-700'}`}>
            Kéo & thả video vào đây
          </p>
          <p className="text-slate-400 text-sm mt-0.5">hoặc nhấp để chọn file</p>
        </div>
        <p className="text-xs text-slate-500">MP4 · AVI · WebM · MOV</p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => pick(e.target.files[0])}
        />
      </div>

      {/* Selected file row */}
      {file && (
        <div className={`${dark ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'} rounded-xl border px-4 py-3 flex flex-wrap items-center gap-3`}>
          <div className="w-9 h-9 bg-slate-100 rounded-lg flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53
                       l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5
                       5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <p className={`text-sm font-medium truncate ${dark ? 'text-slate-100' : 'text-slate-700'}`}>{file.name}</p>
            <p className="text-xs text-slate-400">{formatBytes(file.size)}</p>
          </div>
          {onPreprocessModeChange && (
            <select
              value={preprocessMode}
              onChange={(e) => onPreprocessModeChange(e.target.value)}
              className={`${dark ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-white border-slate-300 text-slate-700'}
                         w-full sm:w-auto text-xs border rounded-lg px-2.5 py-2 focus:outline-none focus:border-blue-500`}
            >
              {PREPROCESS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          )}
          {onOcrBackendChange && (
            <select
              value={ocrBackend}
              onChange={(e) => onOcrBackendChange(e.target.value)}
              className={`${dark ? 'bg-slate-900 border-slate-700 text-slate-100' : 'bg-white border-slate-300 text-slate-700'}
                         w-full sm:w-auto text-xs border rounded-lg px-2.5 py-2 focus:outline-none focus:border-blue-500`}
            >
              {OCR_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={start}
            disabled={loading}
            className="w-full sm:w-auto justify-center bg-blue-600 hover:bg-blue-700 disabled:opacity-60 text-white text-sm
                       font-semibold px-5 py-2 rounded-lg transition-colors flex items-center gap-2"
          >
            {loading ? (
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
            {loading ? 'Đang tải…' : 'Phân tích'}
          </button>
        </div>
      )}
    </div>
  )
}
