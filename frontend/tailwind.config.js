/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "outline-variant": "#94a3b8",
        "tertiary-container": "#fce7f3",
        "primary-fixed-dim": "#a78bfa",
        "on-tertiary-fixed-variant": "#831843",
        "on-error-container": "#7f1d1d",
        "surface-bright": "#ffffff",
        "on-surface-variant": "#64748b",
        "surface-container-high": "#e2e8f0",
        "on-secondary-fixed-variant": "#064e3b",
        "secondary-container": "#d1fae5",
        "on-tertiary-fixed": "#4c0519",
        "tertiary-fixed-dim": "#f9a8d4",
        "surface-tint": "#8b5cf6",
        "primary": "#8b5cf6", // Vibrant Violet, excellent on dark and light
        "on-secondary-container": "#065f46",
        "primary-fixed": "#ede9fe",
        "on-surface": "#0f172a",
        "on-tertiary": "#ffffff",
        "background": "#f8fafc",
        "on-primary-fixed-variant": "#5b21b6",
        "surface": "#ffffff",
        "on-primary": "#ffffff",
        "surface-container": "#f1f5f9",
        "primary-container": "#6d28d9",
        "surface-container-highest": "#cbd5e1",
        "surface-dim": "#e2e8f0",
        "tertiary-fixed": "#fbcfe8",
        "on-secondary": "#ffffff",
        "inverse-primary": "#c4b5fd",
        "inverse-surface": "#1e293b",
        "inverse-on-surface": "#f8fafc",
        "on-tertiary-container": "#9d174d",
        "secondary-fixed": "#a7f3d0",
        "error-container": "#fee2e2",
        "on-secondary-fixed": "#022c22",
        "surface-container-low": "#f8fafc",
        "error": "#ef4444",
        "on-primary-fixed": "#2e1065",
        "on-primary-container": "#ede9fe",
        "on-error": "#ffffff",
        "on-background": "#0f172a",
        "secondary": "#10b981",
        "outline": "#64748b",
        "surface-container-lowest": "#ffffff",
        "tertiary": "#ec4899",
        "secondary-fixed-dim": "#6ee7b7",
        "surface-variant": "#f1f5f9"
      },
      borderRadius: {
        "DEFAULT": "1rem",
        "lg": "2rem",
        "xl": "3rem",
        "full": "9999px"
      },
      fontFamily: {
        "headline": ["Outfit"],
        "body": ["Plus Jakarta Sans"],
        "label": ["Plus Jakarta Sans"]
      }
    },
  },
  plugins: [],
}
