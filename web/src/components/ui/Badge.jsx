import { cx } from './utils'

const toneClass = {
  neutral: 'border-[var(--color-border)] bg-white/5 text-[var(--color-text-muted)]',
  info: 'border-cyan-300/25 bg-cyan-300/10 text-cyan-100',
  success: 'border-emerald-300/25 bg-emerald-300/10 text-emerald-100',
  warning: 'border-amber-300/25 bg-amber-300/10 text-amber-100',
  danger: 'border-red-300/25 bg-red-300/10 text-red-100',
}

export default function Badge({ children, tone = 'neutral', className }) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold',
        toneClass[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}
