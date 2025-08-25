import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const USERNAME = process.env.APP_USERNAME!;
  const PASSWORD = process.env.APP_PASSWORD!;
  const LOAD_DATA_KEY = process.env.LOAD_DATA_KEY!;
  const RENDER_API_URL = "https://malaysia-opr-tracking.onrender.com/data/interbank_rates";

  const auth = Buffer.from(`${USERNAME}:${PASSWORD}`).toString('base64');

  try {
    const response = await fetch(RENDER_API_URL, {
      headers: {
        "Authorization": `Basic ${auth}`,
        "x-api-key": LOAD_DATA_KEY
      }
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("Render API error:", text);
      return res.status(response.status).send(text);
    }

    const data = await response.json();
    res.status(200).json(data);

  } catch (err) {
    console.error("Vercel handler error:", err);
    res.status(500).json({ error: "Server error" });
  }
}
