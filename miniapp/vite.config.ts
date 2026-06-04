import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/auth': 'http://aug:8000',
      '/chat': 'http://aug:8000',
      '/threads': 'http://aug:8000',
      '/files': 'http://aug:8000',
      '/settings': 'http://aug:8000',
      '/models': 'http://aug:8000',
      '/health': 'http://aug:8000',
      '/skills': 'http://aug:8000',
      // ws:true proxies the screencast WebSocket upgrade as well as /browser/status.
      '/browser': { target: 'http://aug:8000', ws: true },
    },
  },
})
