import { cx } from './ui'

function toneForConfidence(conf) {
  if (conf >= 0.9) return 'bg-emerald-300/20 text-emerald-100 border-emerald-300/35'
  if (conf >= 0.7) return 'bg-amber-300/20 text-amber-100 border-amber-300/35'
  if (conf > 0) return 'bg-red-300/20 text-red-100 border-red-300/35'
  return 'bg-white/10 text-[var(--color-text-muted)] border-[var(--color-border)]'
}

export default function PlateDisplay({ chars, compact = false }) {
  if (!chars || chars.length === 0) {
    return compact ? null : (
      <span className="text-sm text-[var(--color-text-muted)]">Đang nhận dạng…</span>
    )
  }

  return (
    <div className={cx('flex flex-wrap items-end', compact ? 'gap-1' : 'gap-1.5')}>
      {chars.map(([ch, conf], index) => {
        const displayChar = ch === '#' ? '?' : ch === '[SEP]' ? ' ' : ch
        const tone = toneForConfidence(conf)

        if (compact) {
          return (
            <span
              key={`${displayChar}-${index}`}
              title={`${Math.round(conf * 100)}%`}
              className={cx(
                'plate-font flex h-5 w-4 items-center justify-center rounded border text-[10px] font-bold',
                tone,
              )}
            >
              {displayChar}
            </span>
          )
        }

        return (
          <div key={`${displayChar}-${index}`} className="flex flex-col items-center gap-1" title={`${Math.round(conf * 100)}% tin cậy`}>
            <span className={cx('plate-font flex h-9 w-8 items-center justify-center rounded-md border text-sm font-bold', tone)}>
              {displayChar}
            </span>
            <span className="data-font text-[10px] text-[var(--color-text-subtle)]">
              {Math.round(conf * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
