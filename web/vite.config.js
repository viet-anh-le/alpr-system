import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload':    'http://localhost:8000',
      '/stream':    'http://localhost:8000',
      '/records':   'http://localhost:8000',
      '/monitor':   { target: 'http://localhost:8000', changeOrigin: true },
      '/incidents': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})

