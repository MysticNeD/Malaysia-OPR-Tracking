import React, { useMemo } from 'react';
import { Card } from '@/components/ui/card';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export interface ChartDataItem {
  date: string;           // YYYY-MM-DD
  opr: number;
  interbank_rate: number;
  myor_volume: number;
  interbank_volume: number;
}

interface OPRChartProps {
  oprs: any[];            // 後端 oprs 數組
  myor_volumes: any[];    // 後端 myor volumes
  interbank_rates: any[];  // 後端 interbank rates
  interbank_volumes: any[]; // 後端 interbank volumes
}

export const OPRChart: React.FC<OPRChartProps> = ({
  oprs,
  myor_volumes,
  interbank_rates,
  interbank_volumes,
}) => {
  // 生成过去 12 个月的日期（每月同日）
  const chartData = useMemo(() => {
    const result: ChartDataItem[] = [];
    const today = new Date();
    for (let i = 0; i < 12; i++) {
      const d = new Date(today.getFullYear(), today.getMonth() - i, today.getDate());
      const dateStr = d.toISOString().split('T')[0]; // YYYY-MM-DD

      // 找对应日期的数据
      const oprItem = oprs.find((o) => o.date === dateStr);
      const myorItem = myor_volumes.find((m) => m.date === dateStr);
      const ibRateItem = interbank_rates.find((r) => r.date === dateStr && r.tenor === 'overnight');
      const ibVolItem = interbank_volumes.find((v) => v.date === dateStr);

      result.push({
        date: dateStr,
        opr: oprItem?.new_opr_level ?? 0,
        myor_volume: myorItem?.aggregate_volume ?? 0,
        interbank_rate: ibRateItem?.rate ?? 0,
        interbank_volume: ibVolItem?.volume ?? 0,
      });
    }

    // 逆序让时间从旧到新显示
    return result.reverse();
  }, [oprs, myor_volumes, interbank_rates, interbank_volumes]);

  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card">
      <h3 className="text-xl font-bold text-foreground mb-6">
        Past 12 Months Rates & Volumes
      </h3>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
            
            {/* 左轴：利率 */}
            <YAxis
              yAxisId="left"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              domain={['auto', 'auto']}
              tickFormatter={(value) => `${value.toFixed(2)}%`}
            />
            
            {/* 右轴：volume */}
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />

            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === 'opr' || name === 'interbank_rate') return [`${value.toFixed(2)}%`, name];
                return [value.toLocaleString(), name]; // volume格式化
              }}
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                color: 'hsl(var(--foreground))',
              }}
            />
            <Legend />

            {/* 折线：利率 */}
            <Line yAxisId="left" type="monotone" dataKey="opr" stroke="hsl(var(--primary))" strokeWidth={3} dot={{ r: 3 }} name="OPR" />
            <Line yAxisId="left" type="monotone" dataKey="interbank_rate" stroke="hsl(var(--financial-green))" strokeWidth={2} dot={{ r: 2 }} name="Interbank Rate" />

            {/* 柱状：Volume */}
            <Bar yAxisId="right" dataKey="myor_volume" fill="hsl(var(--financial-orange))" name="MYOR Volume" barSize={20} />
            <Bar yAxisId="right" dataKey="interbank_volume" fill="hsl(var(--muted-foreground))" name="Interbank Volume" barSize={20} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};
