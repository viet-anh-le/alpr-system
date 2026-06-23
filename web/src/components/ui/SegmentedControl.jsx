import { cx } from './utils'

export default function SegmentedControl({ value, onChange, options, className }) {
  return (
    <div
      className={cx(
        'inline-flex flex-wrap items-center gap-1 rounded-[var(--radius-control)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-1',
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => !option.disabled && onChange(option.value)}
            disabled={option.disabled}
            className={cx(
              'min-h-8 rounded-md px-3 text-xs font-semibold transition-colors duration-200',
              active
                ? 'bg-[var(--color-accent)] text-[var(--color-accent-ink)]'
                : 'text-[var(--color-text-muted)] hover:bg-white/5 hover:text-[var(--color-text)]',
              option.disabled && 'cursor-not-allowed opacity-45',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
