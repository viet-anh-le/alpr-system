import useWebRTC from '../../hooks/monitor/useWebRTC'

export default function LiveViewer({ whepUrl, mjpegUrl }) {
  const { videoRef, status, error } = useWebRTC(whepUrl)

  return (
    <div className="bg-black rounded-lg overflow-hidden relative aspect-video">
      {status !== 'error' ? (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-contain"
        />
      ) : (
        <>
          <img src={mjpegUrl} alt="live" className="w-full h-full object-contain" />
          <div className="absolute top-2 left-2 text-xs bg-amber-900/80 text-amber-100 px-2 py-1 rounded">
            WebRTC unavailable — using MJPEG fallback ({error})
          </div>
        </>
      )}
      {status === 'connecting' && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
          Đang kết nối…
        </div>
      )}
    </div>
  )
}
