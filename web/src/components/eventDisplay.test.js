import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { getEventVehicles, getVehicleClusters } from './monitor/eventDisplay.js'

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
})
