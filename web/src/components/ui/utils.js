export function cx(...parts) {
  return parts.flat().filter(Boolean).join(' ')
}
