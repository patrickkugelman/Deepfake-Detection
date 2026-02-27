import { Eye, Camera, AudioWaveform, HeartPulse, Check, AlertTriangle, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AgentStatusProps {
  compact?: boolean;
}

const agents = [
  { 
    name: 'Vision Agent', 
    icon: Eye, 
    status: 'online' as const,
    description: 'Analiză vizuală și detectare artefacte',
    lastActive: '2s ago'
  },
  { 
    name: 'Audio Agent', 
    icon: AudioWaveform, 
    status: 'online' as const,
    description: 'Analiză audio și detecție sinteză vocală',
    lastActive: '5s ago'
  },
  { 
    name: 'Bio Agent', 
    icon: HeartPulse, 
    status: 'online' as const,
    description: 'Semnale fiziologice și micro-expresii',
    lastActive: '1s ago'
  },
];

export function AgentStatus({ compact = false }: AgentStatusProps) {
  const statusStyles = {
    online: 'bg-success',
    offline: 'bg-destructive',
    processing: 'bg-warning animate-pulse',
  };

  const statusIcons = {
    online: Check,
    offline: X,
    processing: AlertTriangle,
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        {agents.map((agent) => (
          <div 
            key={agent.name}
            className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-card border border-border/50"
            title={agent.name}
          >
            <div className={cn("w-2 h-2 rounded-full", statusStyles[agent.status])} />
            <agent.icon className="h-3.5 w-3.5 text-muted-foreground" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {agents.map((agent) => {
        const StatusIcon = statusIcons[agent.status];
        return (
          <div 
            key={agent.name}
            className="glass-card rounded-xl p-4 flex items-center gap-4"
          >
            <div className={cn(
              "p-2.5 rounded-lg",
              agent.status === 'online' ? 'bg-success/10' : 'bg-destructive/10'
            )}>
              <agent.icon className={cn(
                "h-5 w-5",
                agent.status === 'online' ? 'text-success' : 'text-destructive'
              )} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">{agent.name}</span>
                <div className={cn("w-2 h-2 rounded-full", statusStyles[agent.status])} />
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">{agent.description}</p>
            </div>
            <span className="text-xs text-muted-foreground">{agent.lastActive}</span>
          </div>
        );
      })}
    </div>
  );
}
