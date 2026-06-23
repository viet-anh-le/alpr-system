import { cx } from './utils'

export default function IconButton({ label, children, className, variant = 'secondary', ...props }) {
  const variants = {
    secondary: 'border-[var(--color-border)] bg-[var(--color-panel)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-panel-soft)]',
    ghost: 'border-transparent text-[var(--color-text-muted)] hover:bg-white/5 hover:text-[var(--color-text)]',
    danger: 'border-red-400/30 bg-red-500/10 text-red-100 hover:bg-red-500/20',
  }

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cx(
        'inline-flex h-10 w-10 items-center justify-center rounded-[var(--radius-control)] border transition-colors duration-200',
        variants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}
