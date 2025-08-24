import React, { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { ProgressRing } from '@/components/ui/progress-ring';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

type CardType = 'increase' | 'decrease' | 'hold';
type ProbKey = 'up' | 'down' | 'same';

const typeToKey: Record<CardType, ProbKey> = {
  increase: 'up',
  decrease: 'down',
  hold: 'same',
};

interface ProbabilityCardProps {
  type: CardType;
  title: string;
  /** 直接传百分比(0-100)。若提供，则本组件不请求后端 */
  probability?: number;
  /** 可选：显示“较上次 +x%”之类的字样 */
  change?: string;
  /** 可选：自拉数据时使用的 API。默认 /predict?next_only=true */
  endpoint?: string;
  /** 可选：自拉数据时指定日期 */
  date?: string;
}

export const ProbabilityCard: React.FC<ProbabilityCardProps> = ({
  type,
  title,
  probability,
  change,
  endpoint,
  date,
}) => {
  const [autoProb, setAutoProb] = useState<number | null>(null); // 百分比 0-100
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // 只有当没有传入 probability 时才触发拉取
  useEffect(() => {
    if (probability !== undefined) return;

    const url =
      endpoint ??
      (date
        ? `http://127.0.0.1:8000/predict?date=${encodeURIComponent(date)}`
        : 'http://127.0.0.1:8000/predict?next_only=true');

    const abort = new AbortController();
    setLoading(true);
    setErr(null);

    fetch(url, { signal: abort.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json) => {
        const arr = Array.isArray(json) ? json : json?.data;
        if (!Array.isArray(arr) || arr.length === 0) throw new Error('Empty response');

        const key = typeToKey[type];
        const p = arr[0]?.probabilities?.[key];
        if (typeof p !== 'number') throw new Error('Invalid payload');

        setAutoProb(p * 100); // 转成百分比
      })
      .catch((e: any) => {
        if (e.name === 'AbortError') return;
        console.error('ProbabilityCard fetch error:', e);
        setErr(e.message || 'Fetch error');
      })
      .finally(() => setLoading(false));

    return () => abort.abort();
  }, [probability, endpoint, date, type]);

  const getIcon = () => {
    switch (type) {
      case 'increase':
        return <TrendingUp className="h-8 w-8 text-financial-red" />;
      case 'decrease':
        return <TrendingDown className="h-8 w-8 text-financial-green" />;
      case 'hold':
        return <Minus className="h-8 w-8 text-financial-orange" />;
    }
  };

  const getColor = () => {
    switch (type) {
      case 'increase':
        return 'text-financial-red';
      case 'decrease':
        return 'text-financial-green';
      case 'hold':
        return 'text-financial-orange';
    }
  };

  const value = probability ?? autoProb ?? 0; // 最终用于渲染的百分比

  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card transition-all duration-300 hover:shadow-glow">
      <div className="flex flex-col items-center space-y-4">
        {getIcon()}
        <h3 className="text-lg font-semibold text-foreground text-center">
          {title}
        </h3>

        <ProgressRing value={value} size="lg">
          <div className="text-center">
            <div className={`text-2xl font-bold ${getColor()}`}>
              {loading ? '...' : `${value.toFixed(1)}%`}
            </div>
            {err && (
              <div className="text-xs text-red-500 mt-1">Failed: {err}</div>
            )}
            {!err && change && (
              <div className="text-sm text-muted-foreground mt-1">
                {change}
              </div>
            )}
          </div>
        </ProgressRing>
      </div>
    </Card>
  );
};
