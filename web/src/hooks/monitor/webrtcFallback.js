export const WEBRTC_FALLBACK_TIMEOUT_MS = 9000

export function shouldFallbackToMjpeg(iceConnectionState) {
  return !['connected', 'completed'].includes(iceConnectionState)
}
