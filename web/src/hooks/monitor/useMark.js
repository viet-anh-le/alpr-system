import { API_BASE } from '../../apiClient'

export async function postMark(sessionId, body) {
  const resp = await fetch(`${API_BASE}/monitor/${sessionId}/mark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new Error(await resp.text())
  return (await resp.json()).event_id
}
