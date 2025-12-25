/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'magentic': {
          primary: '#8B5CF6',
          secondary: '#EC4899',
          dark: '#1F2937',
        }
      }
    },
  },
  plugins: [],
}
