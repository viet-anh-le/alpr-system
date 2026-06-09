import IncidentCard from './IncidentCard'

export default function IncidentsPanel({ incidents }) {
  const list = Object.values(incidents).sort(
    (a, b) => new Date(b.markedAt) - new Date(a.markedAt),
  )
  return (
    <div className="w-80 flex-shrink-0 bg-slate-900/30 rounded-lg p-3 overflow-y-auto"
         style={{ height: 'calc(100vh - 200px)' }}>
      <div className="text-xs text-slate-400 mb-3">
        Sự cố ({list.length})
      </div>
      {list.length === 0 ? (
        <div className="text-xs text-slate-500 text-center py-8">
          Chưa có sự cố nào.
        </div>
      ) : (
        list.map((inc) => <IncidentCard key={inc.id} incident={inc} />)
      )}
    </div>
  )
}
