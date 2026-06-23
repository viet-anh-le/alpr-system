import { cx } from './utils'

const variantClass = {
  primary:
    'border border-cyan-300/30 bg-[var(--color-accent)] text-[var(--color-accent-ink)] hover:bg-cyan-200 disabled:bg-cyan-300/40 disabled:text-black/50',
  secondary:
    'border border-[var(--color-border)] bg-[var(--color-panel)] text-[var(--color-text)] hover:border-[var(--color-border-strong)] hover:bg-[var(--color-panel-soft)]',
  ghost:
    'border border-transparent text-[var(--color-text-muted)] hover:bg-white/5 hover:text-[var(--color-text)]',
  danger:
    'border border-red-400/40 bg-red-500/15 text-red-100 hover:bg-red-500/25',
  success:
    'border border-emerald-300/30 bg-emerald-400/15 text-emerald-100 hover:bg-emerald-400/25',
}

const sizeClass = {
  sm: 'min-h-9 px-3 text-xs',
  md: 'min-h-10 px-4 text-sm',
  lg: 'min-h-12 px-5 text-sm',
}

export default function Button({
  children,
  className,
  variant = 'secondary',
  size = 'md',
  loading = false,
  fullWidth = false,
  disabled,
  type = 'button',
  ...props
}) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-[var(--radius-control)] font-semibold transition-colors duration-200',
        'disabled:cursor-not-allowed disabled:opacity-70',
        fullWidth && 'w-full',
        variantClass[variant],
        sizeClass[size],
        className,
      )}
      {...props}
    >
      {loading && (
        <span className="h-4 w-4 rounded-full border-2 border-current border-t-transparent animate-spin" />
      )}
      {children}
    </button>
  )
}
