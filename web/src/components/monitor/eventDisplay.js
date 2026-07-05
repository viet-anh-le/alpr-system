function recognitionId(vehicle) {
  return Number(vehicle?.recognition_id ?? vehicle?.track_id ?? vehicle?.id ?? 0)
}

export function getEventVehicles(event) {
  return Object.values(event?.vehicles || {}).sort(
    (a, b) => recognitionId(b) - recognitionId(a),
  )
}

export function getVehicleClusters(vehicle) {
  return Array.isArray(vehicle?.clusters) ? vehicle.clusters : []
}
