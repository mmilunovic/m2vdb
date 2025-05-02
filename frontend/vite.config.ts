import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['plotly.js-dist-min']
  },
  build: {
    commonjsOptions: {
      include: [/plotly\.js-dist-min/, /node_modules/]
    }
  }
})
