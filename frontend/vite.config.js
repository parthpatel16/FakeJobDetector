import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict-text': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/predict-image': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/predict-url': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
