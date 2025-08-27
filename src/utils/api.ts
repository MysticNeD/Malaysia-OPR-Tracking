// src/utils/api.ts
const API_BASE_URL = import.meta.env.VITE_DATA_THE_API_ABCDE?.replace(/\/?$/, "/");
const API_KEY = import.meta.env.VITE_LOAD_DATA_KEY;

export async function apiFetch(path: string) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "x-api-key": API_KEY,
    },
  });

  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`API returned non-JSON response (status ${res.status})`);
  }

  if (!res.ok) {
    // 明确抛出后端错误，而不是把对象丢给 .map()
    throw new Error(data?.detail || `API error ${res.status}`);
  }

  return data;
}
