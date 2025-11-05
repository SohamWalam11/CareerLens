import js from "@eslint/js";
import tsParser from "@typescript-eslint/parser";
import tseslint from "@typescript-eslint/eslint-plugin";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  {
    ignores: ["dist", "node_modules"],
  },
  js.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"] ,
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true }
      }
    },
    // Assume browser runtime for TS/TSX files so DOM globals (window, document,
    // HTMLInputElement, crypto, setTimeout, console, etc.) are recognized by
    // ESLint and do not trigger `no-undef`.
    env: {
      browser: true,
      es2021: true,
    },
    // Project defines some build-time globals (Vite define) like __API_BASE_URL__.
    globals: {
      __API_BASE_URL__: "readonly",
    },
    plugins: {
      "@typescript-eslint": tseslint,
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh
    },
    rules: {
      ...tseslint.configs.recommended.rules,
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }]
    }
  }
  ,
  // Node-specific files (build config) should allow node globals like `process`.
  {
    files: ["vite.config.*", "**/vite.config.*"],
    env: { node: true },
    languageOptions: {
      parserOptions: { sourceType: "module", ecmaVersion: "latest" }
    }
  }
];
