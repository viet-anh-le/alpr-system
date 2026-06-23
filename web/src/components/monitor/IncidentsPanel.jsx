import { Badge, EmptyState } from '../ui'
import IncidentCard from './IncidentCard'

export default function IncidentsPanel({ incidents }) {
  const list = Object.values(incidents).sort(
    (a, b) => new Date(b.markedAt) - new Date(a.markedAt),
  )

  return (
    <section className="surface-panel flex min-h-[520px] flex-col overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Incident queue</p>
          <h2 className="mt-1 text-lg font-bold">Sự cố đã đánh dấu</h2>
        </div>
        <Badge tone="info">{list.length} marks</Badge>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {list.length === 0 ? (
          <EmptyState title="Chưa có sự cố">
            Mark live stream hoặc chọn interval trong video upload để chạy ALPR trên cửa sổ ngắn.
          </EmptyState>
        ) : (
          <div className="space-y-3">
            {list.map((incident) => <IncidentCard key={incident.id} incident={incident} />)}
          </div>
        )}
      </div>
    </section>
  )
}
