import { Field } from './Field'
import { cx } from './utils'

export default function Select({ label, description, error, options, className, ...props }) {
  return (
    <Field label={label} description={description} error={error}>
      <select
        className={cx(
          'min-h-10 w-full rounded-[var(--radius-control)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3 text-sm text-[var(--color-text)]',
          'transition-colors duration-200 hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]',
          className,
        )}
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  )
}
