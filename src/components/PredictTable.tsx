import React, { useEffect, useState } from "react";

interface Prediction {
  date: string;
  probability: number;
}

const PredictTable: React.FC = () => {
  const [data, setData] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/predict") // 调用 FastAPI 后端
      .then((res) => res.json())
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <p className="text-center text-gray-500">Loading predictions...</p>;
  }

  return (
    <div className="overflow-x-auto p-4">
      <table className="min-w-full border border-gray-300 rounded-lg shadow-md">
        <thead className="bg-gray-200">
          <tr>
            <th className="px-4 py-2 text-left">Date</th>
            <th className="px-4 py-2 text-left">Probability</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item, idx) => (
            <tr key={idx} className="border-t">
              <td className="px-4 py-2">{item.date}</td>
              <td className="px-4 py-2">{(item.probability * 100).toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default PredictTable;
