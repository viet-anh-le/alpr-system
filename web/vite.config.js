import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload':    'http://localhost:8000',
      '/stream':    'http://localhost:8000',
      '/records':   'http://localhost:8000',
      '/auth':      'http://localhost:8000',
      '/sessions':  'http://localhost:8000',
      '/jobs':      'http://localhost:8000',
      '/monitor':   { target: 'http://localhost:8000', changeOrigin: true },
      '/events': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
