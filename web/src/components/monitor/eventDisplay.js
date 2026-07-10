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

export function buildEventInspectionVehicle(vehicle, cluster = null) {
  if (!vehicle) return null
  if (!cluster) {
    return {
      ...vehicle,
      ocr_vote_summary: vehicle.ocr_vote_summary || vehicle.vote_summary || {},
    }
  }

  const clusterPlate = cluster.plate_text ?? cluster.plate
  const clusterFrames = Array.isArray(cluster.track_buffer)
    ? cluster.track_buffer
    : vehicle.track_buffer
  const clusterVotes = cluster.ocr_vote_summary || cluster.vote_summary || {}

  return {
    ...vehicle,
    ...cluster,
    id: vehicle.id,
    recognition_id: vehicle.recognition_id ?? vehicle.track_id ?? vehicle.id,
    cluster_index: cluster.cluster_index,
    plate: clusterPlate ?? vehicle.plate_text ?? vehicle.plate,
    plate_text: clusterPlate ?? vehicle.plate_text ?? vehicle.plate,
    chars: cluster.chars || vehicle.chars || [],
    vehicle_b64: vehicle.vehicle_b64,
    vehicle_image_url: vehicle.vehicle_image_url,
    vehicle_thumbnail_url: vehicle.vehicle_thumbnail_url,
    plate_b64: cluster.plate_b64 ?? vehicle.plate_b64,
    plate_image_url: cluster.plate_image_url ?? vehicle.plate_image_url,
    best_plate_frame: cluster.best_plate_frame ?? vehicle.best_plate_frame,
    track_buffer: clusterFrames,
    ocr_vote_summary: clusterVotes,
    vote_summary: clusterVotes,
    ocr_method: cluster.ocr_method ?? vehicle.ocr_method,
    ocr_frames: cluster.ocr_frames ?? cluster.frame_count ?? vehicle.ocr_frames,
    frame_count: cluster.frame_count ?? cluster.ocr_frames ?? vehicle.frame_count,
    plate_text_confidence:
      cluster.plate_text_confidence ??
      cluster.confidence ??
      vehicle.plate_text_confidence ??
      vehicle.confidence,
    confidence:
      cluster.confidence ??
      cluster.plate_text_confidence ??
      vehicle.confidence ??
      vehicle.plate_text_confidence,
  }
}
