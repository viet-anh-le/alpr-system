import { cx } from './utils'

export function Field({ label, description, error, children, className }) {
  return (
    <label className={cx('block space-y-1.5', className)}>
      {label && <span className="text-xs font-semibold text-[var(--color-text-muted)]">{label}</span>}
      {children}
      {description && !error && (
        <span className="block text-xs text-[var(--color-text-subtle)]">{description}</span>
      )}
      {error && <span className="block text-xs text-red-200">{error}</span>}
    </label>
  )
}

export function TextInput({ className, ...props }) {
  return (
    <input
      className={cx(
        'min-h-10 w-full rounded-[var(--radius-control)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 text-sm text-[var(--color-text)]',
        'placeholder:text-[var(--color-text-subtle)] transition-colors duration-200 hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]',
        className,
      )}
      {...props}
    />
  )
}
