import { API_BASE } from './apiUrl'

export { API_BASE, resolveApiUrl } from './apiUrl'

let csrfToken = null

export function resetCsrfToken() {
  csrfToken = null
}

async function ensureCsrfToken() {
  if (csrfToken) return csrfToken
  const res = await fetch(`${API_BASE}/auth/csrf`, { 
    credentials: 'include',
    headers: {
      'ngrok-skip-browser-warning': '1'
    }
  })
  if (!res.ok) throw new Error('Không lấy được CSRF token')
  const data = await res.json()
  csrfToken = data.csrf_token
  return csrfToken
}

export async function apiFetch(path, options = {}) {
  const method = (options.method || 'GET').toUpperCase()
  const headers = new Headers(options.headers || {})
  const needsCsrf = !['GET', 'HEAD', 'OPTIONS'].includes(method) && options.csrf !== false

  if (needsCsrf) {
    headers.set('X-CSRF-Token', await ensureCsrfToken())
  }
  
  // Bỏ qua trang cảnh báo của ngrok nếu đang dùng ngrok url
  headers.set('ngrok-skip-browser-warning', '1')

  return fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'include',
  })
}

export async function apiJson(path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (options.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const res = await apiFetch(path, { ...options, headers })
  let data = null
  const text = await res.text()
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { detail: text }
    }
  }
  if (!res.ok) {
    const message = data?.detail || data?.error || `HTTP ${res.status}`
    throw new Error(message)
  }
  return data
}
