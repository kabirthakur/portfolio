import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// https://vitejs.dev/config/
export default defineConfig({
  base: '/portfolio/',
  plugins: [react()],
  server: {
    // Open the app in the default browser whenever you run `npm run dev`
    open: '/portfolio/',
  },
})