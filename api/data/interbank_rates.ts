import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const USERNAME = process.env.APP_USERNAME;
  const PASSWORD = process.env.APP_PASSWORD;
  const RENDER_API_URL = "https://malaysia-opr-tracking.onrender.com/data/interbank_rates";

  const auth = Buffer.from(`${USERNAME}:${PASSWORD}`).toString('base64');

  try {
    const response = await fetch(RENDER_API_URL, {
      headers: {
        "Authorization": `Basic ${auth}`
      }
    });
    const data = await response.json();
    res.status(200).json(data);
  } catch (err) {
    console.error("Error fetching Render API:", err);
    res.status(500).json({ error: "Failed to fetch data" });
  }
}
