/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    chunkSizeWarningLimit: 800,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'https://ai-medical-assistant-production-38a3.up.railway.app',
        changeOrigin: true,
      },
      '/hubs': {
        target: 'https://ai-medical-assistant-production-38a3.up.railway.app',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    include: ['src/**/*.{test,spec}.{ts,tsx}', 'src/integration/**/*.{test,spec}.{ts,tsx}'],
    css: true,
  },
})