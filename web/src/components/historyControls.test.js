import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  ALL_SESSIONS,
  buildRecordsPath,
  buildSessionsPath,
  displayPlateText,
  normalizeHistorySummary,
  pageInfo,
} from './historyControls.js'

describe('history controls helpers', () => {
  it('builds a paginated global records query with plate and vehicle filters', () => {
    const path = buildRecordsPath({
      page: 3,
      limit: 12,
      sessionId: ALL_SESSIONS,
      plate: '  30a  ',
      vehicleClass: 'car',
    })

    assert.equal(path, '/records?limit=12&offset=24&plate=30a&vehicle_class=car')
  })

  it('builds a session-scoped records query', () => {
    const path = buildRecordsPath({
      page: 2,
      limit: 10,
      sessionId: 'job-1',
    })

    assert.equal(path, '/records?limit=10&offset=10&session_id=job-1')
  })

  it('builds a paginated sessions query', () => {
    assert.equal(buildSessionsPath({ page: 4, limit: 20 }), '/sessions?limit=20&offset=60')
  })

  it('reports page ranges and boundaries', () => {
    assert.deepEqual(pageInfo(42, 2, 12), {
      total: 42,
      page: 2,
      totalPages: 4,
      start: 13,
      end: 24,
      hasPrev: true,
      hasNext: true,
    })
  })

  it('normalizes summary plate values for display and hover titles', () => {
    const summary = normalizeHistorySummary({
      total_records: 3,
      unique_plates: 2,
      top_plates: [
        { plate_text: '30A[SEP]12345', count: 2, avg_confidence: 0.9 },
      ],
    })

    assert.equal(displayPlateText('51F[SEP]99999'), '51F 99999')
    assert.equal(summary.totalRecords, 3)
    assert.equal(summary.uniquePlates, 2)
    assert.deepEqual(summary.topPlates, [
      { plateText: '30A 12345', count: 2, avgConfidence: 0.9 },
    ])
  })
})
