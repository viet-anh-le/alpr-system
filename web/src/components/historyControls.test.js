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
import { VEHICLE_LABEL } from './workbench/constants.js'

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

  it('encodes vehicle filters whose canonical class names contain spaces', () => {
    const path = buildRecordsPath({
      page: 1,
      limit: 12,
      sessionId: 'job-1',
      vehicleClass: 'delivery tricycle',
    })

    assert.equal(path, '/records?limit=12&offset=0&session_id=job-1&vehicle_class=delivery+tricycle')
  })

  it('offers all six YOLOv5 vehicle filters with user-facing labels', () => {
    const values = HISTORY_VEHICLE_FILTER_OPTIONS.map((option) => option.value)
    const labelsByValue = Object.fromEntries(
      HISTORY_VEHICLE_FILTER_OPTIONS.map((option) => [option.value, option.label]),
    )

    assert.deepEqual(values, [
      'all',
      'car',
      'motorbike',
      'bus',
      'truck',
      'van',
      'delivery tricycle',
    ])
    assert.equal(labelsByValue.motorbike, 'Motorbike')
    assert.equal(labelsByValue.van, 'Xe van')
    assert.equal(labelsByValue['delivery tricycle'], 'Xe ba gác')
    assert.equal(values.includes('motorcycle'), false)
    assert.equal(values.includes('motorbike_rider'), false)
  })

  it('labels all six YOLOv5 vehicle classes across result views', () => {
    assert.equal(VEHICLE_LABEL.car, 'Ô tô')
    assert.equal(VEHICLE_LABEL.motorbike, 'Motorbike')
    assert.equal(VEHICLE_LABEL.bus, 'Xe buýt')
    assert.equal(VEHICLE_LABEL.truck, 'Xe tải')
    assert.equal(VEHICLE_LABEL.van, 'Xe van')
    assert.equal(VEHICLE_LABEL['delivery tricycle'], 'Xe ba gác')
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
