import React from 'react';
import { cn } from '@/lib/utils';

interface ProgressRingProps {
  value: number;
  size?: 'sm' | 'md' | 'lg';
  children?: React.ReactNode;
  className?: string;
  strokeWidth?: number;
}

export const ProgressRing: React.FC<ProgressRingProps> = ({
  value,
  size = 'md',
  children,
  className,
  strokeWidth = 8
}) => {
  const sizeMap = {
    sm: 80,
    md: 120,
    lg: 160
  };

  const dimension = sizeMap[size];
  const radius = (dimension - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg
        width={dimension}
        height={dimension}
        className="transform -rotate-90"
      >
        <circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke="hsl(var(--border))"
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        <circle
          cx={dimension / 2}
          cy={dimension / 2}
          r={radius}
          stroke="hsl(var(--primary))"
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-in-out"
        />
      </svg>
      {children && (
        <div className="absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      )}
    </div>
  );
};