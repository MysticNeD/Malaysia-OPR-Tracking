import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  
  console.log("Vercel Function sees LOAD_DATA_KEY as:", process.env.LOAD_DATA_KEY);
  const RENDER_API_URL = "https://malaysia-opr-tracking.onrender.com/predict";

  try {
    const response = await fetch(RENDER_API_URL, {
      method: "GET",
      headers: {
        "x-api-key": process.env.LOAD_DATA_KEY!,
        "Authorization": "Basic " + Buffer.from(
          `${process.env.APP_USERNAME}:${process.env.APP_PASSWORD}`
        ).toString("base64"),
      },
    });

    if (!response.ok) {
      // 非 200 响应，返回原始文本
      const text = await response.text();
      console.error("Render API returned error:", text);
      return res.status(response.status).json({ error: text });
    }

    // 尝试解析 JSON
    let data;
    try {
      data = await response.json();
    } catch (jsonErr) {
      const text = await response.text();
      console.error("Failed to parse JSON:", text);
      return res.status(500).send({ error: "Render API returned invalid JSON", raw: text });
    }

    // 确认返回是数组
    if (!Array.isArray(data)) {
      console.warn("Render API did not return an array:", data);
      return res.status(500).json({ error: "Invalid data format", raw: data });
    }

    res.status(200).json(data);

  } catch (err) {
    console.error("Vercel serverless handler error:", err);
    res.status(500).json({ error: "Server error" });
  }
}
