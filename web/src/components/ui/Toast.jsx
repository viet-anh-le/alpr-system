import { useEffect } from 'react'
import IconButton from './IconButton'

export default function Toast({ message, tone = 'danger', onDismiss }) {
  useEffect(() => {
    if (!message) return undefined
    const timeout = window.setTimeout(onDismiss, 7000)
    return () => window.clearTimeout(timeout)
  }, [message, onDismiss])

  if (!message) return null

  const toneClass = {
    danger: 'border-red-300/30 bg-red-500/15 text-red-50',
    info: 'border-cyan-300/30 bg-cyan-300/15 text-cyan-50',
    success: 'border-emerald-300/30 bg-emerald-300/15 text-emerald-50',
  }[tone]

  return (
    <div className={`fixed bottom-5 left-1/2 z-50 flex max-w-[min(92vw,720px)] -translate-x-1/2 items-center gap-3 rounded-xl border px-4 py-3 text-sm shadow-[var(--shadow-panel)] backdrop-blur ${toneClass}`}>
      <span className="h-2 w-2 flex-none rounded-full bg-current" />
      <span className="min-w-0 flex-1">{message}</span>
      <IconButton label="Đóng thông báo" variant="ghost" className="h-8 w-8" onClick={onDismiss}>
        <span aria-hidden>×</span>
      </IconButton>
    </div>
  )
}
