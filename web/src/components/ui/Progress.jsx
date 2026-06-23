import { cx } from './utils'

export default function Progress({ value = 0, tone = 'info', className }) {
  const clamped = Math.max(0, Math.min(100, Number(value) || 0))
  const tones = {
    info: 'bg-[var(--color-accent)]',
    success: 'bg-[var(--color-success)]',
    warning: 'bg-[var(--color-warning)]',
    danger: 'bg-[var(--color-danger)]',
  }

  return (
    <div className={cx('h-2 overflow-hidden rounded-full bg-black/30', className)}>
      <div
        className={cx('h-full rounded-full transition-all duration-300', tones[tone])}
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}
