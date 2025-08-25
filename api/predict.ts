import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const response = await fetch("https://malaysia-opr-tracking.onrender.com/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Basic " + Buffer.from(
        process.env.APP_USERNAME + ":" + process.env.APP_PASSWORD
      ).toString("base64")
    },
    body: req.body
  });

  const data = await response.json();
  res.status(200).json(data);
}
