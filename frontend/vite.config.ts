import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: { environment: 'jsdom' },
  server: {
    proxy: { '/analizar': 'http://127.0.0.1:8000' },
  },
})
