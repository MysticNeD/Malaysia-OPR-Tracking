import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const RENDER_API_URL = "https://malaysia-opr-tracking.onrender.com/data/interbank_rates"; // Adjust URL
  try {
    const response = await fetch(RENDER_API_URL, {
      method: "GET",
      headers: {
        "x-api-key": process.env.LOAD_DATA_KEY!,
        "Authorization": "Basic " + Buffer.from(`${process.env.APP_USERNAME}:${process.env.APP_PASSWORD}`).toString("base64"),
      },
    });
    if (!response.ok) {
      const text = await response.text();
      return res.status(response.status).json({ error: text }); // Use json() for proper Content-Type
    }
    const data = await response.json();
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
}