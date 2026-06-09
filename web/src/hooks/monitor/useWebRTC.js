import { useEffect, useRef, useState } from 'react'

/**
 * useWebRTC — minimal WHEP client.
 * Returns { videoRef, status, error } and attaches the inbound track
 * to videoRef.current automatically.
 */
export default function useWebRTC(whepUrl) {
  const videoRef = useRef(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!whepUrl) return undefined
    let pc = new RTCPeerConnection()
    let cancelled = false

    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })

    pc.ontrack = (e) => {
      if (videoRef.current) videoRef.current.srcObject = e.streams[0]
    }
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'connected') setStatus('live')
      if (pc.iceConnectionState === 'failed') {
        setStatus('error')
        setError('WebRTC ICE failed')
      }
    }

    setStatus('connecting')
    ;(async () => {
      try {
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        const resp = await fetch(whepUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp,
        })
        if (!resp.ok) throw new Error('WHEP POST failed: ' + resp.status)
        const answerSdp = await resp.text()
        if (cancelled) return
        await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
      } catch (e) {
        if (!cancelled) {
          setStatus('error')
          setError(e.message)
        }
      }
    })()

    return () => {
      cancelled = true
      pc.close()
      pc = null
    }
  }, [whepUrl])

  return { videoRef, status, error }
}
