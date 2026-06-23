import Button from './Button'

export default function EmptyState({ title, children, actionLabel, onAction }) {
  return (
    <div className="flex min-h-48 flex-col items-center justify-center rounded-[var(--radius-panel)] border border-dashed border-[var(--color-border)] bg-black/10 p-6 text-center">
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--color-border)] bg-white/5 text-[var(--color-accent)]">
        <span aria-hidden>⌁</span>
      </div>
      <h3 className="text-sm font-bold text-[var(--color-text)]">{title}</h3>
      {children && <p className="mt-2 max-w-md text-sm text-[var(--color-text-muted)]">{children}</p>}
      {actionLabel && onAction && (
        <Button className="mt-4" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  )
}
