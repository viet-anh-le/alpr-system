import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { resolveApiUrl } from './apiUrl.js'

describe('resolveApiUrl', () => {
  it('keeps relative backend URLs relative when no API base is configured', () => {
    assert.equal(resolveApiUrl('/monitor/upload/abc/video', ''), '/monitor/upload/abc/video')
  })

  it('prefixes relative backend URLs with the deployed API base', () => {
    assert.equal(
      resolveApiUrl('/monitor/upload/abc/video', 'https://pod-8000.proxy.runpod.net/'),
      'https://pod-8000.proxy.runpod.net/monitor/upload/abc/video',
    )
  })

  it('leaves absolute URLs untouched', () => {
    assert.equal(
      resolveApiUrl('https://pod-8889.proxy.runpod.net/live/whep', 'https://api.example.com'),
      'https://pod-8889.proxy.runpod.net/live/whep',
    )
  })
})
