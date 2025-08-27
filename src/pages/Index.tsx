//Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
// npm run dev
import React, { useEffect, useState } from 'react';
import { ProbabilityCard } from '@/components/ProbabilityCard';
import { MetricCard } from '@/components/MetricCard';
import { MPCCountdown } from '@/components/MPCCountdown';
import { OPRChart } from '@/components/OPRChart';
import PredictTable from '@/components/PredictTable';
import { 
  Percent, 
  TrendingUp, 
  DollarSign, 
  Activity,
  Building2,
  Globe
} from 'lucide-react';
import dashboardBg from '@/assets/dashboard-bg.jpg';
import { apiFetch } from "@/utils/api";

function escapeHtml(unsafe: string) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


const Index = () => {
  // 状态：用来存后端数据
  const [probabilities, setProbabilities] = useState({
    up: 0,
    down: 0,
    same: 0
  });
  const [currentOPR, setCurrentOPR] = useState<string>("--");
  const [oprs, setOprs] = useState<any[]>([]);
  const [myor_vol, setMyorVol] = useState<any[]>([]);
  const [ib_rate, setIbRate] = useState<any[]>([]);
  const [ib_vol, setIbVol] = useState<any[]>([]);

  useEffect(() => {
    apiFetch("predict")
      .then(data => {
        console.log("Predict data:", data);
        if (Array.isArray(data) && data.length > 0) {
          const item = data[0];  // 取第一个预测对象
          setProbabilities({
            up: item.probabilities.up * 100,
            same: item.probabilities.same * 100,
            down: item.probabilities.down * 100,
          });
          setCurrentOPR(item.predicted_opr + "%");
        } else {
          console.warn("Predict API returned empty array or invalid data:", data);
        }
      })
      .catch(err => console.error("API fetch error (predict):", err));
  }, []);


  useEffect(() => {
    apiFetch("data/oprs")
      .then(data => {
        console.log("✅ API raw data:", data);   // 👈 先打印出来看看
        if (Array.isArray(data) && data.length > 0) {
          console.log("👉 Latest OPR:", data[data.length - 1]); // 打印最后一个
          setOprs(data);  
        } else {
          console.warn("⚠️ API 返回的不是数组或者为空:", data);
        }
      })
      .catch(err => console.error("API fetch error (oprs):", err));
  }, []);

  useEffect(() => {
    apiFetch("data/myor")
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          console.log("👉 Latest OPR:", data[data.length - 1]); // 打印最后一个
          setMyorVol(data);  
        } else {
          console.warn("⚠️ API 返回的不是数组或者为空:", data);
        }
      })
      .catch(err => console.error("API fetch error (oprs):", err));
  }, []);

  useEffect(() => {
    apiFetch("data/interbank_rates")
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          console.log("👉 Latest IBR:", data[data.length - 1]); // 打印最后一个
          setIbRate(data);  
        } else {
          console.warn("⚠️ API 返回的不是数组或者为空:", data);
        }
      })
      .catch(err => console.error("API fetch error (oprs):", err));
  }, []);

  useEffect(() => {
    apiFetch("data/interbank_volumes")
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          console.log("👉 Latest IBV:", data[data.length - 1]); // 打印最后一个
          setIbVol(data);  
        } else {
          console.warn("⚠️ API 返回的不是数组或者为空:", data);
        }
      })
      .catch(err => console.error("API fetch error (oprs):", err));
  }, []);

  const latestOpr = oprs[oprs.length - 1];
  const latestMyorVol = myor_vol[myor_vol.length - 1];
  const latestIbRate = ib_rate.find(item => item.tenor === 'overnight');
  const latestIbVol = ib_vol.find(item => item.tenor === 'overnight');

  const metrics = [
    {
      title: 'Current OPR',
      value: latestOpr ? `${escapeHtml(String(latestOpr.new_opr_level))}%` : "--",
      change: ' ',
      changeType: 'neutral' as const,
      icon: Percent,
      description: 'Bank Negara Malaysia Overnight Policy Rate'
    },

    {
      title: 'MYOR Aggrerate Volume',
      value: latestMyorVol ? `${escapeHtml(String(latestMyorVol.aggregate_volume))}` : "--",
      change: 'Updated Every day',
      changeType: 'positive' as const,
      icon: TrendingUp,
      description: 'MY0R-i Daily Volume'
    },
    {
      title: 'Kuala Lumpur Interbank - Rate',
      value: latestIbRate ?.tenor === "overnight" ? `${escapeHtml(String(latestIbRate.rate))}` : "--",
      change: 'Updated every day',
      changeType: 'positive' as const,
      icon: Activity,
      description: 'Kuala Lumpur Interbank Offered Rate'
    },
    {
      title: 'Kuala Lumpur Interbank - Volume',
      value: latestIbVol ?.tenor === "overnight" ? `${escapeHtml(String(latestIbVol.volume))}` : "--",
      change: 'Updated every day',
      changeType: 'negative' as const,
      icon: Activity,
      description: 'Kuala Lumpur Interbank Daily Volume'
    }
  ];

interface ChartDataItem {
    date: string;
    opr: number;
    myor_volume: number;
    interbank_rate: number;
    interbank_volume: number;
  }

  const chartData: ChartDataItem[] = oprs.map(oprItem => {
    const myorItem = myor_vol.find(item => item.date === oprItem.date);
    const ibRateItem = ib_rate.find(item => item.date === oprItem.date && item.tenor === 'overnight');
    const ibVolItem = ib_vol.find(item => item.date === oprItem.date);

    return {
      date: oprItem.date,
      opr: oprItem.new_opr_level,
      myor_volume: myorItem?.aggregate_volume ?? 0,
      interbank_rate: ibRateItem?.rate ?? 0,
      interbank_volume: ibVolItem?.volume ?? 0
    };
  });




  return (
    <div 
      className="min-h-screen bg-background relative"
      style={{
        backgroundImage: `linear-gradient(rgba(34, 37, 61, 0.8), rgba(34, 37, 61, 0.9)), url(${dashboardBg})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed'
      }}
    >
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-white 200 mb-4 bg-gradient-primary bg-clip-text">
            Malaysia OPR Tracker
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            This website provides real-time Malaysia's Overnight Policy Rate and using financial metrics to predict the possibility of next OPR movement.
            The current OPR is {latestOpr ? `${escapeHtml(String(latestOpr.new_opr_level))}%` : 'Loading...'}.
          </p>

          <p className="text-sm text-muted-foreground max-w-2xl mx-auto mt-4">
            Note: If the data are not showing up, please wait for a minute to load the data. If the problem persists, please reload the page and 
            the data should be showing up. (Sorry because the backend server is free and will sleep after a period of inactivity 🥲.)
          </p>

        </div>

        {/* MPC Countdown */}
        <div className="mb-8">
          <MPCCountdown />
        </div>

        {/* Probability Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <ProbabilityCard
            type="increase"
            probability={probabilities.up}
            title="Rate Increase"
            change="+0.25% Expected"
          />
          <ProbabilityCard
            type="hold"
            probability={probabilities.same}
            title="Rate Hold"
            change="Maintain Current"
          />
          <ProbabilityCard
            type="decrease"
            probability={probabilities.down}
            title="Rate Decrease"
            change="-0.25% Expected"
          />
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {metrics.map((metric, index) => (
            <MetricCard
              key={index}
              title={metric.title}
              value={metric.value}
              change={metric.change}
              changeType={metric.changeType}
              icon={metric.icon}
              description={metric.description}
            />
          ))}
        </div>
        {/* Historical Chart */}
    

        {/* Prediction Table */}
        <div className="mb-8">
          <PredictTable />
        </div>

        {/* Footer */}
        <div className="text-center">
          <p className="text-sm text-muted-foreground">
            Disclaimer: This is only a demo project and for educational purposes only.
            This is not a financial advice and should not be used for any research or trading decisions.
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Created by Wei Zhe (Bryan) Chong @ 2025
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            Last updated: {new Date().toLocaleString('en-MY')}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Index;
