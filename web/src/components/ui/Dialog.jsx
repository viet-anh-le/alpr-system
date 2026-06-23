import { useEffect } from 'react'
import IconButton from './IconButton'
import { cx } from './utils'

export default function Dialog({ open, onClose, title, description, children, className }) {
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm animate-fade-in" onMouseDown={onClose}>
      <section
        className={cx('surface-panel max-h-[88vh] w-full max-w-3xl overflow-hidden animate-zoom-in', className)}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="panel-header">
          <div className="min-w-0">
            <h2 className="text-base font-bold">{title}</h2>
            {description && <p className="mt-1 text-sm text-[var(--color-text-muted)]">{description}</p>}
          </div>
          <IconButton label="Đóng" variant="ghost" onClick={onClose}>
            <span aria-hidden className="text-xl leading-none">×</span>
          </IconButton>
        </div>
        {children}
      </section>
    </div>
  )
}
