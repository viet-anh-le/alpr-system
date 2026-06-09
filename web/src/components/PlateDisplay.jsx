/**
 * compact=false (default): large blocks with % label — used in full grid view
 * compact=true: small blocks, no % label — used inside the side panel cards
 */
export default function PlateDisplay({ chars, compact = false }) {
  if (!chars || chars.length === 0) {
    return compact ? null : (
      <span className="text-sm text-slate-400 italic">Đang nhận dạng…</span>
    )
  }

  return (
    <div className={`flex flex-wrap items-end ${compact ? 'gap-0.5' : 'gap-1'}`}>
      {chars.map(([ch, conf], i) => {
        const displayChar = ch === '#' ? '?' : ch === '[SEP]' ? ' ' : ch
        const bg =
          conf >= 0.90 ? 'bg-green-600' :
          conf >= 0.70 ? 'bg-amber-500' :
          conf >  0    ? 'bg-red-500'   : 'bg-slate-400'

        if (compact) {
          return (
            <span
              key={i}
              title={`${Math.round(conf * 100)}%`}
              className={`plate-font ${bg} text-white text-[10px] font-bold
                          w-4 h-5 flex items-center justify-center rounded-sm`}
            >
              {displayChar}
            </span>
          )
        }

        return (
          <div
            key={i}
            className="flex flex-col items-center gap-0.5"
            title={`${Math.round(conf * 100)}% tin cậy`}
          >
            <span
              className={`plate-font ${bg} text-white text-sm font-bold
                          w-7 h-8 flex items-center justify-center rounded`}
            >
              {displayChar}
            </span>
            <span className="text-[9px] text-slate-400 tabular-nums">
              {Math.round(conf * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}
