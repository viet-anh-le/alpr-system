import { Badge, EmptyState } from '../ui'
import useWebRTC from '../../hooks/monitor/useWebRTC'

export default function LiveViewer({ whepUrl, mjpegUrl }) {
  const { videoRef, status, error } = useWebRTC(whepUrl)
  const statusLabel = getLiveStatusLabel(status)

  return (
    <section className="surface-panel overflow-hidden">
      <div className="panel-header">
        <div>
          <p className="section-label">Quan sát trực tiếp</p>
          <h2 className="mt-1 text-lg font-bold">Luồng camera</h2>
        </div>
        <Badge tone={status === 'error' ? 'warning' : status === 'connecting' ? 'info' : 'success'}>
          {statusLabel}
        </Badge>
      </div>
      <div className="relative bg-black">
        {status !== 'error' ? (
          <video ref={videoRef} autoPlay muted playsInline className="aspect-video w-full object-contain" />
        ) : (
          <>
            <img src={mjpegUrl} alt="luồng camera dự phòng" className="aspect-video w-full object-contain" />
            <div className="absolute left-3 top-3 rounded-lg border border-amber-300/30 bg-amber-300/15 px-3 py-2 text-xs text-amber-50">
              Không dùng được WebRTC, đang dùng MJPEG dự phòng: {error}
            </div>
          </>
        )}
        {status === 'connecting' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/45">
            <EmptyState title="Đang kết nối camera">Đợi WHEP/WebRTC thiết lập luồng.</EmptyState>
          </div>
        )}
      </div>
    </section>
  )
}

function getLiveStatusLabel(status) {
  if (status === 'error') return 'Dự phòng MJPEG'
  if (status === 'connecting') return 'Đang kết nối'
  if (status === 'live') return 'Đang phát'
  return 'Sẵn sàng'
}
