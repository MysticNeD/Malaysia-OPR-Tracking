// src/utils/api.ts
const API_BASE_URL = import.meta.env.VITE_DATA_THE_API_ABCDE?.replace(/\/?$/, "/");
const API_KEY = import.meta.env.VITE_LOAD_API_KEY;

export async function apiFetch(path: string) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "x-api-key": API_KEY,
    },
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }

  return res.json();
}
