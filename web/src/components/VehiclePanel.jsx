import VehicleCard from './VehicleCard'

export default function VehiclePanel({ vehicles, totalDone, jobId }) {
  const list = [...vehicles].reverse() // newest first

  return (
    <div className="flex flex-col bg-slate-800 rounded-2xl overflow-hidden shadow-lg h-full">
      {/* Header — mimics the reference screenshot style */}
      <div className="bg-blue-700 px-4 py-2.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          {/* Camera icon */}
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24"
               stroke="currentColor" strokeWidth={2}>
            <rect x="2" y="7" width="20" height="14" rx="2"
                  stroke="currentColor" fill="none" />
            <circle cx="12" cy="13" r="3" fill="currentColor" opacity={0.7} />
          </svg>
          <span className="text-white text-xs font-bold uppercase tracking-widest">
            Phát hiện phương tiện
          </span>
        </div>
        <span className="bg-blue-600 text-white text-xs font-bold px-2 py-0.5 rounded-full">
          {vehicles.length}
        </span>
      </div>

      {/* Cards list — scrollable */}
      <div className="flex-1 overflow-y-auto">
        {list.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-500 py-12">
            <svg className="w-10 h-10 opacity-30" fill="none" viewBox="0 0 24 24"
                 stroke="currentColor" strokeWidth={1.2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                    d="M9 17.25v1.007a3 3 0 01-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0115
                       18.257V17.25m6-12V15a2.25 2.25 0 01-2.25 2.25H5.25A2.25 2.25 0 013
                       15V5.25m18 0A2.25 2.25 0 0018.75 3H5.25A2.25 2.25 0 003 5.25m18 0H3" />
            </svg>
            <p className="text-sm">Chưa phát hiện xe nào</p>
            <p className="text-xs text-slate-600">Upload video để bắt đầu phân tích</p>
          </div>
        ) : (
          list.map(v => <VehicleCard key={v.id} vehicle={v} jobId={jobId} />)
        )}
      </div>

      {/* Footer stats */}
      {vehicles.length > 0 && (
        <div className="flex-shrink-0 border-t border-slate-700 px-3 py-2
                        flex items-center justify-between text-[10px] text-slate-400">
          <span>Tổng: <strong className="text-slate-200">{vehicles.length}</strong> xe</span>
          <span>Đã xác nhận: <strong className="text-emerald-400">{totalDone}</strong></span>
        </div>
      )}
    </div>
  )
}
