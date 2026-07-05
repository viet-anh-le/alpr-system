let lockDepth = 0
let previousOverflow = ''

export function lockBodyScroll() {
  if (typeof document === 'undefined') return () => {}

  if (lockDepth === 0) {
    previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
  }
  lockDepth += 1

  return () => {
    lockDepth = Math.max(0, lockDepth - 1)
    if (lockDepth === 0) {
      document.body.style.overflow = previousOverflow
      previousOverflow = ''
    }
  }
}
