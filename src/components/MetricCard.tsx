import React from 'react';
import { Card } from '@/components/ui/card';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: LucideIcon;
  description?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  description
}) => {
  const getChangeColor = () => {
    switch (changeType) {
      case 'positive':
        return 'text-financial-green';
      case 'negative':
        return 'text-financial-red';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <Card className="p-6 bg-gradient-card border-border shadow-card transition-all duration-300 hover:shadow-glow">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <Icon className="h-5 w-5 text-primary" />
            <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold text-foreground">{value}</div>
            {change && (
              <div className={`text-sm ${getChangeColor()} flex items-center mt-1`}>
                {change}
              </div>
            )}
            {description && (
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};