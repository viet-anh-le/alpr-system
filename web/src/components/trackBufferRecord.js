export function getTrackRecordId(vehicle) {
  return vehicle?.recognition_id ?? vehicle?.id ?? vehicle?.track_id
}

export function buildInlineRecord(vehicle, jobId) {
  const frames = Array.isArray(vehicle?.track_buffer) ? vehicle.track_buffer : []
  if (!vehicle || frames.length === 0) return null

  const bestFrame = vehicle.best_plate_frame || frames.reduce(
    (best, frame) => (frameScore(frame) > frameScore(best) ? frame : best),
    null,
  )
  const charConf = averageCharConfidence(vehicle.chars)
  const fallbackConfidence = Number(vehicle.plate_text_confidence ?? vehicle.confidence)

  return {
    session_id: vehicle.session_id ?? jobId,
    track_id: getTrackRecordId(vehicle),
    cluster_index: numberOrUndefined(vehicle.cluster_index),
    vehicle_track_id: vehicle.vehicle_track_id,
    plate_track_id: vehicle.plate_track_id,
    vehicle_class: vehicle.vehicle_class ?? vehicle.cls,
    vehicle_thumbnail_url: vehicle.vehicle_thumbnail_url ?? vehicle.vehicle_b64,
    best_plate_frame: bestFrame || {
      frame_index: null,
      quality_score: vehicle.confidence || 0,
      image_b64: vehicle.plate_b64,
      image_url: vehicle.plate_image_url,
    },
    track_buffer: frames,
    plate_text: vehicle.plate_text ?? vehicle.plate,
    chars: vehicle.chars || [],
    plate_text_confidence: Number.isFinite(fallbackConfidence)
      ? fallbackConfidence
      : charConf,
    ocr_vote_summary: vehicle.ocr_vote_summary || vehicle.vote_summary || {},
    ocr_method: vehicle.ocr_method || 'realtime_buffer',
  }
}

export function mergePersistedRecord(record, inlineRecord) {
  if (!inlineRecord) return normalizePersistedRecord(record)
  if (inlineRecord.cluster_index !== undefined) {
    const cluster = findPersistedCluster(record, inlineRecord.cluster_index)
    return cluster
      ? normalizePersistedCluster(record, cluster, inlineRecord)
      : inlineRecord
  }

  const normalized = normalizePersistedRecord(record)
  const hasPersistedFrames = Array.isArray(normalized?.track_buffer)
    && normalized.track_buffer.length > 0
  if (hasPersistedFrames) return normalized

  return {
    ...inlineRecord,
    ...normalized,
    best_plate_frame: normalized?.best_plate_frame || inlineRecord.best_plate_frame,
    track_buffer: inlineRecord.track_buffer,
    ocr_vote_summary: normalized?.ocr_vote_summary || inlineRecord.ocr_vote_summary,
  }
}

export function frameScore(frame) {
  if (!frame) return -1
  if (Number.isFinite(Number(frame.combined_score))) return Number(frame.combined_score)
  const quality = Number(frame.quality_score) || 0
  const ocrConfidence = Math.max(Number(frame.ocr_confidence) || 0.1, 0.1)
  return quality * ocrConfidence
}

function normalizePersistedRecord(record) {
  if (!record) return record
  return {
    ...record,
    track_id: record.track_id ?? getTrackRecordId(record),
    vehicle_class: record.vehicle_class ?? record.cls,
    plate_text: record.plate_text ?? record.plate,
    chars: record.chars || [],
    ocr_vote_summary: record.ocr_vote_summary || record.vote_summary || {},
  }
}

function normalizePersistedCluster(parent, cluster, inlineRecord) {
  const confidence = Number(cluster.plate_text_confidence ?? cluster.confidence)
  return {
    ...inlineRecord,
    ...cluster,
    session_id: parent?.session_id ?? inlineRecord.session_id,
    track_id: parent?.track_id ?? inlineRecord.track_id,
    cluster_index: inlineRecord.cluster_index,
    vehicle_track_id: parent?.vehicle_track_id ?? inlineRecord.vehicle_track_id,
    plate_track_id: parent?.plate_track_id ?? inlineRecord.plate_track_id,
    vehicle_class: parent?.vehicle_class ?? inlineRecord.vehicle_class,
    vehicle_thumbnail_url: parent?.vehicle_thumbnail_url ?? inlineRecord.vehicle_thumbnail_url,
    plate_text: cluster.plate_text ?? cluster.plate ?? inlineRecord.plate_text,
    chars: cluster.chars || inlineRecord.chars,
    best_plate_frame: cluster.best_plate_frame || inlineRecord.best_plate_frame,
    track_buffer: Array.isArray(cluster.track_buffer) ? cluster.track_buffer : inlineRecord.track_buffer,
    plate_text_confidence: Number.isFinite(confidence)
      ? confidence
      : inlineRecord.plate_text_confidence,
    ocr_vote_summary: cluster.ocr_vote_summary || cluster.vote_summary || inlineRecord.ocr_vote_summary,
    ocr_method: cluster.ocr_method || parent?.ocr_method || inlineRecord.ocr_method,
  }
}

function findPersistedCluster(record, clusterIndex) {
  if (!Array.isArray(record?.clusters)) return null
  return record.clusters.find(
    (cluster) => Number(cluster.cluster_index) === Number(clusterIndex),
  ) || null
}

function averageCharConfidence(chars) {
  if (!Array.isArray(chars) || chars.length === 0) return 0
  return chars.reduce((sum, item) => sum + (Number(item?.[1]) || 0), 0) / chars.length
}

function numberOrUndefined(value) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : undefined
}
