/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DATA_THE_API_ABCDE: string;
  readonly VITE_LOAD_API_KEY: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}