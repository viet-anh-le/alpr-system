import { useState } from 'react'
import DropZone from '../DropZone'

const OCR_OPTIONS = [
  { value: 'default', label: 'Mặc định' },
  { value: 'smalllpr_ctc', label: 'SmallLPR CTC' },
  { value: 'parseq', label: 'PARSeq' },
  { value: 'yolov5_char', label: 'YOLOv5 Char' },
]

export default function SourceSelector({ onConnectLive, onSelectFile }) {
  const [tab, setTab] = useState('rtsp')
  const [url, setUrl] = useState('')
  const [preprocessMode, setPreprocessMode] = useState('none')
  const [ocrBackend, setOcrBackend] = useState('default')

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
      <div className="flex items-center gap-1 mb-3">
        <button
          onClick={() => setTab('rtsp')}
          className={`text-xs px-3 py-1.5 rounded ${
            tab === 'rtsp' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
          }`}
        >
          RTSP camera
        </button>
        <button
          onClick={() => setTab('upload')}
          className={`text-xs px-3 py-1.5 rounded ${
            tab === 'upload' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
          }`}
        >
          Upload video
        </button>
      </div>

      {tab === 'rtsp' ? (
        <form
          onSubmit={(e) => { e.preventDefault(); if (url.trim()) onConnectLive(url.trim(), ocrBackend) }}
          className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2"
        >
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="rtsp://10.0.0.5:554/main"
            className="flex-1 bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm
                       focus:border-blue-500 focus:outline-none text-white"
          />
          <select
            value={ocrBackend}
            onChange={(e) => setOcrBackend(e.target.value)}
            className="bg-slate-900 border-slate-700 text-slate-100 w-full sm:w-auto text-xs border rounded px-2.5 py-2 focus:outline-none focus:border-blue-500"
          >
            {OCR_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <button
            type="submit"
            className="text-xs px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium whitespace-nowrap"
          >
            Kết nối
          </button>
        </form>
      ) : (
        <DropZone
          onFileSelect={onSelectFile}
          dark
          preprocessMode={preprocessMode}
          onPreprocessModeChange={setPreprocessMode}
          ocrBackend={ocrBackend}
          onOcrBackendChange={setOcrBackend}
        />
      )}
    </div>
  )
}
