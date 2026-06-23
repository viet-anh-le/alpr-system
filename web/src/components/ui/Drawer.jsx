import { useEffect } from 'react'
import IconButton from './IconButton'
import { cx } from './utils'

export default function Drawer({ open, onClose, title, description, children, className }) {
  useEffect(() => {
    if (!open) return undefined
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = previous
      window.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 bg-black/55 p-0 backdrop-blur-sm animate-fade-in" onMouseDown={onClose}>
      <aside
        className={cx(
          'ml-auto flex h-full w-full max-w-5xl flex-col border-l border-[var(--color-border)] bg-[var(--color-bg)] shadow-[var(--shadow-panel)] animate-slide-up',
          className,
        )}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="panel-header bg-[var(--color-bg-elevated)]">
          <div className="min-w-0">
            <h2 className="text-base font-bold text-[var(--color-text)]">{title}</h2>
            {description && <p className="mt-1 text-sm text-[var(--color-text-muted)]">{description}</p>}
          </div>
          <IconButton label="Đóng" variant="ghost" onClick={onClose}>
            <span aria-hidden className="text-xl leading-none">×</span>
          </IconButton>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </aside>
    </div>
  )
}
