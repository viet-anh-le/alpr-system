export const PREPROCESS_OPTIONS = [
  { value: 'none', label: 'Không tiền xử lý' },
  { value: 'night', label: 'Ban đêm' },
  { value: 'low_contrast', label: 'Tương phản thấp' },
  { value: 'fog', label: 'Sương mù' },
  { value: 'rain', label: 'Mưa / nhiễu' },
  { value: 'glare', label: 'Chói sáng' },
]

export const OCR_OPTIONS = [
  { value: 'default', label: 'SmallLPR-Line-CTC (mặc định)' },
  { value: 'vietnamese_yolov5', label: 'YOLOv5 Việt Nam' },
]

export const VEHICLE_LABEL = {
  vehicle: 'Phương tiện',
  car: 'Ô tô',
  motorbike: 'Motorbike',
  bus: 'Xe buýt',
  truck: 'Xe tải',
  van: 'Xe van',
  'delivery tricycle': 'Xe ba gác',
  delivery_tricycle: 'Xe ba gác',
  motorcycle: 'Xe máy',
  motorbike_rider: 'Xe máy',
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '0 KB'
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '--:--'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function cleanPlateText(text) {
  return (text || '').replaceAll('[SEP]', ' ').replaceAll('#', '').trim()
}

export function averageConfidence(chars, fallback = 0) {
  if (Number.isFinite(fallback) && fallback > 0) return Math.round(fallback * 100)
  if (!chars?.length) return 0
  return Math.round(chars.reduce((sum, [, conf]) => sum + conf, 0) / chars.length * 100)
}

export function confidenceTone(value) {
  if (value >= 90) return 'success'
  if (value >= 70) return 'warning'
  if (value > 0) return 'danger'
  return 'neutral'
}
