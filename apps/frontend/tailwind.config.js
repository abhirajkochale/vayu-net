/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        vayu: {
          dark: '#0a0f1d',
          card: '#121a2f',
          border: '#1f2b48',
          accent: '#00d2ff',
          warning: '#ffb300',
          danger: '#ff3366',
        }
      }
    },
  },
  plugins: [],
}
