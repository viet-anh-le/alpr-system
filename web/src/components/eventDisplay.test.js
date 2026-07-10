import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  buildEventInspectionVehicle,
  getEventVehicles,
  getVehicleClusters,
} from './monitor/eventDisplay.js'

describe('monitor event display helpers', () => {
  it('returns vehicles while the event is still processing', () => {
    const event = {
      status: 'processing',
      vehicles: {
        2: { id: 2, plate: '51F-99999' },
        1: { id: 1, plate: '30G-51827' },
      },
    }

    const vehicles = getEventVehicles(event)

    assert.deepEqual(vehicles.map((vehicle) => vehicle.id), [2, 1])
  })

  it('preserves all OCR clusters for a vehicle', () => {
    const vehicle = {
      id: 7,
      plate: '30G-51827',
      clusters: [
        { cluster_index: 0, plate: '30G-51827' },
        { cluster_index: 1, plate: '51F-99999' },
      ],
    }

    const clusters = getVehicleClusters(vehicle)

    assert.equal(clusters.length, 2)
    assert.deepEqual(clusters.map((cluster) => cluster.plate), ['30G-51827', '51F-99999'])
  })

  it('builds a cluster inspection vehicle with cluster-local buffer evidence', () => {
    const vehicle = {
      id: 7,
      track_id: 7,
      vehicle_b64: 'vehicle-crop',
      plate: '30G-51827',
      track_buffer: [{ frame_index: 1, image_b64: 'parent-frame' }],
      ocr_vote_summary: { '30G-51827': 3 },
    }
    const cluster = {
      cluster_index: 1,
      plate_text: '51F-99999',
      plate_b64: 'cluster-plate',
      track_buffer: [{ frame_index: 8, image_b64: 'cluster-frame' }],
      ocr_vote_summary: { '51F-99999': 2 },
      frame_count: 2,
    }

    const inspectionVehicle = buildEventInspectionVehicle(vehicle, cluster)

    assert.equal(inspectionVehicle.id, 7)
    assert.equal(inspectionVehicle.recognition_id, 7)
    assert.equal(inspectionVehicle.cluster_index, 1)
    assert.equal(inspectionVehicle.plate_text, '51F-99999')
    assert.equal(inspectionVehicle.vehicle_b64, 'vehicle-crop')
    assert.deepEqual(inspectionVehicle.track_buffer.map((frame) => frame.frame_index), [8])
    assert.deepEqual(inspectionVehicle.ocr_vote_summary, { '51F-99999': 2 })
  })
})
