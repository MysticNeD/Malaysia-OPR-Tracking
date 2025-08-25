import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    const response = await fetch("https://malaysia-opr-tracking.onrender.com/predict", {
        method: "GET",
        headers: {
            "x-api-key": process.env.LOAD_DATA_KEY!
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
