import { useEffect } from 'react'

export default function ImageModal({ src, alt, onClose }) {
  // Prevent scrolling when modal is open
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEsc)
    return () => {
      document.body.style.overflow = 'auto'
      window.removeEventListener('keydown', handleEsc)
    }
  }, [onClose])

  if (!src) return null

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="relative max-w-5xl w-full max-h-[90vh] flex flex-col items-center animate-zoom-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button 
          onClick={onClose}
          className="absolute -top-12 right-0 p-2 text-white/70 hover:text-white transition-colors bg-white/10 hover:bg-white/20 rounded-full"
          aria-label="Close"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Image Display */}
        <div className="bg-slate-900 ring-1 ring-white/10 rounded-xl overflow-hidden shadow-2xl">
          <img 
            src={src} 
            alt={alt || 'Enlarged view'} 
            className="max-w-full max-h-[80vh] object-contain select-none shadow-inner"
          />
        </div>

        {/* Caption */}
        {alt && (
          <div className="mt-4 px-4 py-1.5 bg-black/40 backdrop-blur-md rounded-full border border-white/10 shadow-lg">
            <p className="text-white/90 text-xs font-medium tracking-wide">
              {alt}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
