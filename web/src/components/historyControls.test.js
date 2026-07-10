import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  HISTORY_VEHICLE_FILTER_OPTIONS,
  buildRecordsPath,
  buildSessionsPath,
  displayPlateText,
  normalizeHistorySummary,
  pageInfo,
} from './historyControls.js'

describe('history controls helpers', () => {
  it('builds a paginated session records query with plate and motorbike filters', () => {
    const path = buildRecordsPath({
      page: 3,
      limit: 12,
      sessionId: 'job-1',
      plate: '  30a  ',
      vehicleClass: 'motorbike',
    })

    assert.equal(path, '/records?limit=12&offset=24&session_id=job-1&plate=30a&vehicle_class=motorbike')
  })

  it('offers only the canonical motorbike vehicle filter', () => {
    const values = HISTORY_VEHICLE_FILTER_OPTIONS.map((option) => option.value)
    const motorbike = HISTORY_VEHICLE_FILTER_OPTIONS.find((option) => option.value === 'motorbike')

    assert.equal(motorbike?.label, 'Motorbike')
    assert.equal(values.includes('motorbike'), true)
    assert.equal(values.includes('motorcycle'), false)
    assert.equal(values.includes('motorbike_rider'), false)
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

  it('normalizes summary plate values without count or confidence display data', () => {
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
      { plateText: '30A 12345' },
    ])
  })
})
