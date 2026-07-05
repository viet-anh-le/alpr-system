import Dialog from './ui/Dialog'

export default function ImageModal({ src, alt, onClose }) {
  return (
    <Dialog
      open={!!src}
      onClose={onClose}
      title="Ảnh đối chiếu"
      description={alt || 'Ảnh cắt phục vụ kiểm chứng kết quả nhận dạng.'}
      className="max-w-5xl"
    >
      <div className="max-h-[78vh] overflow-auto bg-black p-3">
        {src && (
          <img
            src={src}
            alt={alt || 'Ảnh đối chiếu'}
            className="mx-auto max-h-[72vh] max-w-full select-none object-contain"
          />
        )}
      </div>
    </Dialog>
  )
}
