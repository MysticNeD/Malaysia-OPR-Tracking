import React from 'react';
import { Card } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const historicalData = [
  { date: '2023-01', opr: 2.75, myor: 2.73, klibor1m: 2.85 },
  { date: '2023-03', opr: 3.00, myor: 2.98, klibor1m: 3.10 },
  { date: '2023-05', opr: 3.00, myor: 2.99, klibor1m: 3.15 },
  { date: '2023-07', opr: 3.00, myor: 3.01, klibor1m: 3.20 },
  { date: '2023-09', opr: 3.00, myor: 3.02, klibor1m: 3.18 },
  { date: '2023-11', opr: 3.00, myor: 3.00, klibor1m: 3.22 },
  { date: '2024-01', opr: 3.00, myor: 3.01, klibor1m: 3.25 },
  { date: '2024-03', opr: 3.00, myor: 3.02, klibor1m: 3.28 },
  { date: '2024-05', opr: 3.00, myor: 3.03, klibor1m: 3.30 },
  { date: '2024-07', opr: 3.00, myor: 3.04, klibor1m: 3.32 },
  { date: '2024-09', opr: 3.00, myor: 3.05, klibor1m: 3.35 },
  { date: '2024-11', opr: 3.00, myor: 3.06, klibor1m: 3.38 },
];

export const OPRChart: React.FC = () => {
  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card">
      <h3 className="text-xl font-bold text-foreground mb-6">
        Historical Rates & Market Indicators
      </h3>
      
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={historicalData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="date" 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              domain={['dataMin - 0.1', 'dataMax + 0.1']}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
                color: 'hsl(var(--foreground))'
              }}
              formatter={(value: number, name: string) => [
                `${value.toFixed(2)}%`,
                name === 'opr' ? 'OPR' : name === 'myor' ? 'MYOR' : 'KLIBOR 1M'
              ]}
            />
            <Line 
              type="monotone" 
              dataKey="opr" 
              stroke="hsl(var(--primary))" 
              strokeWidth={3}
              dot={{ fill: 'hsl(var(--primary))', r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="myor" 
              stroke="hsl(var(--financial-green))" 
              strokeWidth={2}
              dot={{ fill: 'hsl(var(--financial-green))', r: 3 }}
            />
            <Line 
              type="monotone" 
              dataKey="klibor1m" 
              stroke="hsl(var(--financial-orange))" 
              strokeWidth={2}
              dot={{ fill: 'hsl(var(--financial-orange))', r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="flex justify-center space-x-6 mt-4">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-primary rounded-full"></div>
          <span className="text-sm text-foreground">OPR</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-financial-green rounded-full"></div>
          <span className="text-sm text-foreground">MYOR</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-financial-orange rounded-full"></div>
          <span className="text-sm text-foreground">KLIBOR 1M</span>
        </div>
      </div>
    </Card>
  );
};