const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {})
  if (options.body) headers.set('Content-Type', 'application/json')
  if (token) headers.set('Authorization', `Bearer ${token}`)
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  } catch {
    throw new ApiError('Cannot reach the API. Check that the backend is running.', 0)
  }
  if (response.status === 204) return null
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status})`
    throw new ApiError(detail, response.status)
  }
  return payload
}

export const api = {
  register: (payload) => request('/api/auth/register', { method: 'POST', body: JSON.stringify(payload) }),
  login: (payload) => request('/api/auth/login', { method: 'POST', body: JSON.stringify(payload) }),
  logout: (token) => request('/api/auth/logout', { method: 'POST', token }),
  me: (token) => request('/api/auth/me', { token }),
  dashboard: (token) => request('/api/analytics/dashboard', { token }),
  profile: (token) => request('/api/biometrics/profile', { token }),
  deleteProfile: (token) => request('/api/biometrics/profile', { method: 'DELETE', token }),
  history: (token) => request('/api/analytics/history?limit=50', { token }),
  timing: (token) => request('/api/analytics/timing?limit=50', { token }),
  modelStatus: (token) => request('/api/model/status', { token }),
  modelMetrics: (token) => request('/api/model/metrics', { token }),
  analyze: (token, payload) => request('/api/typing/analyze', { method: 'POST', token, body: JSON.stringify(payload) }),
  enroll: (token, payload) => request('/api/biometrics/enroll', { method: 'POST', token, body: JSON.stringify(payload) }),
  authenticate: (token, payload) => request('/api/typing/authenticate', { method: 'POST', token, body: JSON.stringify(payload) }),
}
