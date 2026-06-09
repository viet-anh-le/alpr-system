import { useEffect } from 'react'

export default function ErrorToast({ message, onDismiss }) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 6000)
    return () => clearTimeout(t)
  }, [message, onDismiss])

  return (
    <div
      className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-red-600 text-white
                 text-sm px-5 py-3 rounded-xl shadow-lg flex items-center gap-2 z-50"
    >
      <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24"
           stroke="currentColor" strokeWidth={2}>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4m0 4h.01" />
      </svg>
      <span>{message}</span>
      <button
        onClick={onDismiss}
        className="ml-2 text-red-200 hover:text-white"
        aria-label="Đóng"
      >
        ✕
      </button>
    </div>
  )
}
