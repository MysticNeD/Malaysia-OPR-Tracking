import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Calendar, Clock } from 'lucide-react';

export const MPCCountdown: React.FC = () => {
  const [timeLeft, setTimeLeft] = useState({
    days: 0,
    hours: 0,
    minutes: 0,
    seconds: 0
  });

  // Next MPC meeting date (example: January 24, 2025)
  const nextMPCDate = new Date('2025-01-24T09:00:00+08:00');

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      const difference = nextMPCDate.getTime() - now.getTime();

      if (difference > 0) {
        setTimeLeft({
          days: Math.floor(difference / (1000 * 60 * 60 * 24)),
          hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
          minutes: Math.floor((difference / 1000 / 60) % 60),
          seconds: Math.floor((difference / 1000) % 60)
        });
      }
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <Card className="p-6 bg-gradient-primary border-border shadow-glow">
      <div className="flex items-center space-x-3 mb-4">
        <Calendar className="h-6 w-6 text-primary-foreground" />
        <h3 className="text-xl font-bold text-primary-foreground">
          Next MPC Meeting
        </h3>
      </div>
      
      <div className="text-center">
        <div className="text-lg text-primary-foreground mb-2">
          {nextMPCDate.toLocaleDateString('en-MY', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
          })}
        </div>
        
        <div className="grid grid-cols-4 gap-4 mt-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-foreground">
              {timeLeft.days}
            </div>
            <div className="text-sm text-primary-foreground/80">Days</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-foreground">
              {timeLeft.hours}
            </div>
            <div className="text-sm text-primary-foreground/80">Hours</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-foreground">
              {timeLeft.minutes}
            </div>
            <div className="text-sm text-primary-foreground/80">Minutes</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-foreground">
              {timeLeft.seconds}
            </div>
            <div className="text-sm text-primary-foreground/80">Seconds</div>
          </div>
        </div>
      </div>
    </Card>
  );
};