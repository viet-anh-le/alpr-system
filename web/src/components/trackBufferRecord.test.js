import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { buildInlineRecord, mergePersistedRecord } from './trackBufferRecord.js'

const parentVehicle = {
  id: 7,
  cls: 'car',
  plate: '30A-12345',
  chars: [['3', 0.9], ['0', 0.9]],
  plate_b64: 'parent-plate',
  vehicle_b64: 'vehicle',
  track_buffer: [
    { frame_index: 1, quality_score: 0.9, image_b64: 'parent-a' },
    { frame_index: 2, quality_score: 0.8, image_b64: 'parent-b' },
  ],
  vote_summary: { '30A-12345': 2 },
}

const clusterVehicle = {
  ...parentVehicle,
  cluster_index: 1,
  plate: '51F-99999',
  chars: [['5', 0.95], ['1', 0.95]],
  track_buffer: [
    { frame_index: 3, quality_score: 0.7, image_b64: 'cluster-a' },
    { frame_index: 4, quality_score: 0.6, image_b64: 'cluster-b' },
  ],
  vote_summary: { '51F-99999': 2 },
  ocr_frames: 2,
}

describe('track buffer record helpers', () => {
  it('builds inline cluster records from cluster-local frames and votes', () => {
    const record = buildInlineRecord(clusterVehicle, 'job-1')

    assert.equal(record.track_id, 7)
    assert.equal(record.cluster_index, 1)
    assert.equal(record.plate_text, '51F-99999')
    assert.deepEqual(record.track_buffer.map((frame) => frame.frame_index), [3, 4])
    assert.deepEqual(record.ocr_vote_summary, { '51F-99999': 2 })
  })

  it('merges a persisted parent record with the matching cluster only', () => {
    const inlineRecord = buildInlineRecord(clusterVehicle, 'job-1')
    const persisted = {
      ...parentVehicle,
      track_id: 7,
      plate_text: '30A-12345',
      ocr_vote_summary: { '30A-12345': 2 },
      clusters: [
        {
          cluster_index: 0,
          plate_text: '30A-12345',
          track_buffer: [{ frame_index: 1, quality_score: 0.9, image_url: 'persisted-parent' }],
          ocr_vote_summary: { '30A-12345': 2 },
        },
        {
          cluster_index: 1,
          plate_text: '51F-99999',
          track_buffer: [{ frame_index: 4, quality_score: 0.8, image_url: 'persisted-cluster' }],
          ocr_vote_summary: { '51F-99999': 1 },
        },
      ],
    }

    const record = mergePersistedRecord(persisted, inlineRecord)

    assert.equal(record.cluster_index, 1)
    assert.equal(record.plate_text, '51F-99999')
    assert.deepEqual(record.track_buffer.map((frame) => frame.frame_index), [4])
    assert.deepEqual(record.ocr_vote_summary, { '51F-99999': 1 })
  })

  it('keeps inline cluster data when persisted records do not include clusters yet', () => {
    const inlineRecord = buildInlineRecord(clusterVehicle, 'job-1')
    const persisted = {
      track_id: 7,
      plate_text: '30A-12345',
      track_buffer: parentVehicle.track_buffer,
      ocr_vote_summary: { '30A-12345': 2 },
    }

    const record = mergePersistedRecord(persisted, inlineRecord)

    assert.equal(record.cluster_index, 1)
    assert.equal(record.plate_text, '51F-99999')
    assert.deepEqual(record.track_buffer.map((frame) => frame.frame_index), [3, 4])
    assert.deepEqual(record.ocr_vote_summary, { '51F-99999': 2 })
  })
})
