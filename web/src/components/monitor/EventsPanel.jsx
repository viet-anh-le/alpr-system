import { Badge, EmptyState } from '../ui'
import EventCard from './EventCard'

export default function EventsPanel({ events }) {
  const list = Object.values(events).sort(
    (a, b) => new Date(b.markedAt) - new Date(a.markedAt),
  )

  return (
    <section className="surface-panel recognition-panel flex flex-col">
      <div className="panel-header">
        <div>
          <p className="section-label">Hàng đợi sự kiện</p>
          <h2 className="mt-1 text-lg font-bold">Sự kiện đã đánh dấu</h2>
        </div>
        <Badge tone="info">{list.length} lượt đánh dấu</Badge>
      </div>
      <div className="recognition-panel-body p-3">
        {list.length === 0 ? (
          <EmptyState title="Chưa có sự kiện">
            Đánh dấu luồng trực tiếp hoặc chọn đoạn trong video tải lên để chạy ALPR trên cửa sổ ngắn.
          </EmptyState>
        ) : (
          <div className="space-y-3">
            {list.map((event) => <EventCard key={event.id} event={event} />)}
          </div>
        )}
      </div>
    </section>
  )
}
