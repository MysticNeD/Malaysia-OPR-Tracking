import React from 'react';
import { Card } from '@/components/ui/card';
import { ProgressRing } from '@/components/ui/progress-ring';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface ProbabilityCardProps {
  type: 'increase' | 'decrease' | 'hold';
  probability: number;
  title: string;
  change?: string;
}

export const ProbabilityCard: React.FC<ProbabilityCardProps> = ({
  type,
  probability,
  title,
  change
}) => {
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

  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card transition-all duration-300 hover:shadow-glow">
      <div className="flex flex-col items-center space-y-4">
        {getIcon()}
        <h3 className="text-lg font-semibold text-foreground text-center">
          {title}
        </h3>
        <ProgressRing value={probability} size="lg">
          <div className="text-center">
            <div className={`text-2xl font-bold ${getColor()}`}>
              {probability.toFixed(1)}%
            </div>
            {change && (
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