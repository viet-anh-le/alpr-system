import { useState, useEffect } from 'react'
import supabase from '../supabaseClient'

const displayPlateText = (text) => (text || '').replaceAll('[SEP]', ' ')

export default function HistoryModal({ onClose }) {
  const [jobs, setJobs] = useState([])
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [vehicles, setVehicles] = useState([])
  const [loading, setLoading] = useState(true)

  // Prevent background scrolling
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = 'auto' }
  }, [])

  // Fetch Jobs
  useEffect(() => {
    async function fetchJobs() {
      if (!supabase) return setLoading(false)
      const { data, error } = await supabase
        .from('jobs')
        .select('*')
        .order('created_at', { ascending: false })
      
      if (!error && data) {
        setJobs(data)
        if (data.length > 0) setSelectedJobId(data[0].id)
      }
      setLoading(false)
    }
    fetchJobs()
  }, [])

  // Fetch Vehicles when a job is selected
  useEffect(() => {
    async function fetchVehicles() {
      if (!supabase || !selectedJobId) return
      const { data, error } = await supabase
        .from('vehicles')
        .select('*')
        .eq('job_id', selectedJobId)
        .order('tracker_id', { ascending: true })

      if (!error && data) setVehicles(data)
    }
    fetchVehicles()
  }, [selectedJobId])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-6xl h-[85vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/50">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Lịch sử Hệ thống
          </h2>
          <button onClick={onClose} className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-full transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {!supabase ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-6 text-center">
            <svg className="w-16 h-16 mb-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M18.364 5.636a9 9 0 00-12.728 0m12.728 0A9 9 0 0112 21m0-18A9 9 0 003 12m18 0a9 9 0 01-9 9m9-9H3" />
            </svg>
            <p className="text-lg font-medium text-white mb-2">Chưa kết nối CSDL</p>
            <p className="text-sm max-w-md">Vui lòng cung cấp `VITE_SUPABASE_URL` và config backend để xem bằng chứng.</p>
          </div>
        ) : (
          <div className="flex-1 flex min-h-0">
            {/* Sidebar (List of Jobs) */}
            <div className="w-72 border-r border-slate-800 flex flex-col bg-slate-900/50 overflow-y-auto">
              {loading ? (
                <p className="p-4 text-sm text-slate-500">Đang tải...</p>
              ) : jobs.length === 0 ? (
                <p className="p-4 text-sm text-slate-500">Chưa có dữ liệu nào được lưu.</p>
              ) : (
                jobs.map(job => (
                  <button
                    key={job.id}
                    onClick={() => setSelectedJobId(job.id)}
                    className={`text-left px-4 py-3 border-b border-slate-800/50 transition-colors ${selectedJobId === job.id ? 'bg-blue-600/20 border-l-2 border-l-blue-500' : 'hover:bg-slate-800 border-l-2 border-l-transparent'}`}
                  >
                    <p className={`font-medium text-sm truncate ${selectedJobId === job.id ? 'text-blue-400' : 'text-slate-200'}`}>
                      {job.filename}
                    </p>
                    <div className="flex justify-between items-center mt-1">
                      <p className="text-xs text-slate-500">#{job.id}</p>
                      <p className="text-[10px] text-slate-500">{new Date(job.created_at).toLocaleDateString()}</p>
                    </div>
                  </button>
                ))
              )}
            </div>

            {/* Main Area (Vehicles Grid) */}
            <div className="flex-1 overflow-y-auto p-6 bg-slate-950/20">
              {vehicles.length === 0 && selectedJobId ? (
                <p className="text-slate-400 text-sm">Không tìm thấy biển số nào trong phiên này.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {vehicles.map(v => (
                    <div key={v.id} className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden flex flex-col">
                      <div className="flex-1 p-3 flex gap-2 justify-center items-center bg-slate-900">
                         {v.vehicle_image_url ? (
                            <img src={v.vehicle_image_url} alt="Vehicle" className="h-24 w-1/2 object-contain bg-slate-950 rounded-md" />
                         ) : <div className="h-24 w-1/2 bg-slate-950 rounded-md" />}
                         {v.plate_image_url ? (
                            <img src={v.plate_image_url} alt="Plate" className="h-24 w-1/2 object-contain bg-black rounded-md" />
                         ) : <div className="h-24 w-1/2 bg-black rounded-md" />}
                      </div>
                      <div className="p-3 bg-slate-800">
                        <div className="flex justify-between items-start mb-1">
                          <p className="plate-font text-xl font-bold text-white tracking-widest">{displayPlateText(v.plate_text)}</p>
                          <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-slate-700 text-slate-300">
                            {v.class_name}
                          </span>
                        </div>
                        <p className="text-xs text-slate-400">ID: {v.tracker_id} &bull; Độ tin cậy: <span className="text-amber-400">{v.confidence}%</span></p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
