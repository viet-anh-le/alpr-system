import { useEffect, useRef, useState } from 'react'
import {
  shouldFallbackToMjpeg,
  WEBRTC_FALLBACK_TIMEOUT_MS,
} from './webrtcFallback'

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
    let fallbackTimer = null

    const clearFallbackTimer = () => {
      if (fallbackTimer) {
        window.clearTimeout(fallbackTimer)
        fallbackTimer = null
      }
    }

    const switchToMjpegFallback = (message) => {
      if (cancelled) return
      clearFallbackTimer()
      setStatus('error')
      setError(message)
      pc?.close()
    }

    pc.addTransceiver('video', { direction: 'recvonly' })
    pc.addTransceiver('audio', { direction: 'recvonly' })

    pc.ontrack = (e) => {
      if (videoRef.current) videoRef.current.srcObject = e.streams[0]
    }
    pc.oniceconnectionstatechange = () => {
      if (!pc) return
      if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
        clearFallbackTimer()
        setStatus('live')
      }
      if (pc.iceConnectionState === 'failed') {
        switchToMjpegFallback('WebRTC ICE failed')
      }
    }

    setStatus('connecting')
    fallbackTimer = window.setTimeout(() => {
      if (shouldFallbackToMjpeg(pc?.iceConnectionState)) {
        switchToMjpegFallback('WebRTC connection timed out')
      }
    }, WEBRTC_FALLBACK_TIMEOUT_MS)

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
          switchToMjpegFallback(e.message)
        }
      }
    })()

    return () => {
      cancelled = true
      clearFallbackTimer()
      pc?.close()
      pc = null
    }
  }, [whepUrl])

  return { videoRef, status, error }
}
