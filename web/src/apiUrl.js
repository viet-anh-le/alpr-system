const viteEnv = import.meta.env || {}

export function normalizeApiBase(value = '') {
  return value ? value.replace(/\/+$/, '') : ''
}

export const API_BASE = normalizeApiBase(viteEnv.VITE_API_BASE_URL || '')

export function resolveApiUrl(url, apiBase = API_BASE) {
  if (!url || typeof url !== 'string') return url

  try {
    new URL(url)
    return url
  } catch {
    // Relative URLs should stay relative in local Vite proxy mode.
  }

  const base = normalizeApiBase(apiBase)
  if (!base || !url.startsWith('/')) return url
  return `${base}${url}`
}
