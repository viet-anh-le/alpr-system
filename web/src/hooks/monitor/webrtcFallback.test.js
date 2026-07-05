import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  shouldFallbackToMjpeg,
  WEBRTC_FALLBACK_TIMEOUT_MS,
} from './webrtcFallback.js'

describe('WebRTC MJPEG fallback', () => {
  it('uses an 8-10 second fallback timeout', () => {
    assert.ok(WEBRTC_FALLBACK_TIMEOUT_MS >= 8000)
    assert.ok(WEBRTC_FALLBACK_TIMEOUT_MS <= 10000)
  })

  it('falls back while ICE has not connected', () => {
    assert.equal(shouldFallbackToMjpeg('new'), true)
    assert.equal(shouldFallbackToMjpeg('checking'), true)
    assert.equal(shouldFallbackToMjpeg('failed'), true)
    assert.equal(shouldFallbackToMjpeg('closed'), true)
  })

  it('does not fall back after ICE connects', () => {
    assert.equal(shouldFallbackToMjpeg('connected'), false)
    assert.equal(shouldFallbackToMjpeg('completed'), false)
  })
})
