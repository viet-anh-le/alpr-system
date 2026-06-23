let csrfToken = null

export function resetCsrfToken() {
  csrfToken = null
}

async function ensureCsrfToken() {
  if (csrfToken) return csrfToken
  const res = await fetch('/auth/csrf', { credentials: 'include' })
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

  return fetch(path, {
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
