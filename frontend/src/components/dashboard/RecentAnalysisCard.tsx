import { HistoryRecord } from '@/api';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Eye, Clock, FileVideo, ChevronRight, Loader2, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface RecentAnalysisCardProps {
  history: HistoryRecord[];
  activeJobs?: Map<string, number>;
}

export function RecentAnalysisCard({ history, activeJobs }: RecentAnalysisCardProps) {
  const navigate = useNavigate();

  const verdictStyles: Record<string, { badge: string; icon: JSX.Element }> = {
    REAL: { 
      badge: 'bg-green-500/10 text-green-500 border-green-500/30',
      icon: <CheckCircle2 className="h-3 w-3" />
    },
    FAKE: { 
      badge: 'bg-red-500/10 text-red-500 border-red-500/30',
      icon: <XCircle className="h-3 w-3" />
    },
    SUSPICIOUS: {
      badge: 'bg-orange-500/10 text-orange-500 border-orange-500/30',
      icon: <AlertCircle className="h-3 w-3" />
    },
    INCONCLUSIVE: { 
      badge: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30',
      icon: <AlertCircle className="h-3 w-3" />
    },
    PENDING: {
      badge: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
      icon: <Loader2 className="h-3 w-3 animate-spin" />
    },
    PROCESSING: {
      badge: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
      icon: <Loader2 className="h-3 w-3 animate-spin" />
    },
    QUEUED: {
      badge: 'bg-gray-500/10 text-gray-500 border-gray-500/30',
      icon: <Clock className="h-3 w-3" />
    },
    ERROR: {
      badge: 'bg-gray-500/10 text-gray-500 border-gray-500/30',
      icon: <XCircle className="h-3 w-3" />
    },
    FAILED: {
      badge: 'bg-gray-500/10 text-gray-500 border-gray-500/30',
      icon: <XCircle className="h-3 w-3" />
    }
  };

  const getVerdictStyle = (verdict: string) => {
    return verdictStyles[verdict] || verdictStyles.INCONCLUSIVE;
  };

  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`;
  };

  const formatTimestamp = (timestamp: string | number) => {
    const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  // Helper: Get unique identifier (job_id preferred, fallback to id if exists)
  const getRecordId = (record: HistoryRecord): string => {
    return record.job_id || (record as any).id || '';
  };

  return (
    <Card className="border-primary/20 bg-muted/5">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-mono flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" />
          RECENT_JOBS // LOGS
          {activeJobs && activeJobs.size > 0 && (
            <Badge variant="outline" className="ml-2 border-blue-500 text-blue-500 text-[10px]">
              {activeJobs.size} ACTIVE
            </Badge>
          )}
        </CardTitle>
        <Button variant="ghost" size="sm" onClick={() => navigate('/history')} className="text-xs">
          View All <ChevronRight className="h-3 w-3 ml-1" />
        </Button>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-border/50">
          {history.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground text-sm">
              <FileVideo className="h-12 w-12 mx-auto mb-2 opacity-20" />
              <p>No recent activity logs found.</p>
              <p className="text-xs mt-1">Upload a file to start analysis.</p>
            </div>
          ) : (
            history.map((record) => {
              const recordId = getRecordId(record);
              const isActive = activeJobs?.has(recordId);
              const progress = activeJobs?.get(recordId) || 0;
              const verdictStyle = getVerdictStyle(record.verdict || record.status || 'PENDING');
              
              return (
                <div 
                  key={recordId} 
                  className={cn(
                    "flex flex-col p-3 transition-colors",
                    isActive ? "bg-blue-500/5" : "hover:bg-muted/50"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <div className={cn(
                        "bg-background p-2 rounded border",
                        isActive && "border-blue-500/50"
                      )}>
                        {isActive ? (
                          <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
                        ) : (
                          <FileVideo className="h-4 w-4 text-muted-foreground" />
                        )}
                      </div>
                      
                      <div className="flex flex-col flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold truncate max-w-[150px] md:max-w-[250px]">
                            {record.filename}
                          </span>
                          {record.confidence > 0 && !isActive && (
                            <Badge variant="outline" className="text-[9px] px-1 py-0">
                              {formatConfidence(record.confidence)}
                            </Badge>
                          )}
                        </div>
                        <span className="text-[10px] text-muted-foreground font-mono">
                          {formatTimestamp(record.timestamp || record.created_at || new Date().toISOString())}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <Badge 
                        variant="outline" 
                        className={cn("text-[10px] flex items-center gap-1", verdictStyle.badge)}
                      >
                        {verdictStyle.icon}
                        {isActive ? `${progress}%` : (record.verdict || record.status || 'PENDING')}
                      </Badge>
                      
                      {!isActive && (
                        <Button 
                          size="icon" 
                          variant="ghost" 
                          className="h-6 w-6" 
                          onClick={() => navigate(`/results/${recordId}`)}
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {isActive && (
                    <div className="mt-3 space-y-1">
                      <Progress value={progress} className="h-1" />
                      <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
                        <span>
                          {progress < 30 && '⏳ Initializing...'}
                          {progress >= 30 && progress < 50 && '🔍 Selecting agents...'}
                          {progress >= 50 && progress < 65 && '🧠 Running neural analysis...'}
                          {progress >= 65 && progress < 80 && '🤖 Gemini Vision processing...'}
                          {progress >= 80 && progress < 90 && '🔬 Neuro-symbolic reasoning...'}
                          {progress >= 90 && progress < 100 && '💾 Saving results...'}
                          {progress >= 100 && '✅ Completed'}
                        </span>
                        <span className="font-bold">{progress}%</span>
                      </div>
                    </div>
                  )}

                  {/* Status Message */}
                  {!isActive && record.verdict === 'FAKE' && (
                    <div className="mt-2 flex items-center gap-2 text-[10px] text-red-500/70">
                      <AlertCircle className="h-3 w-3" />
                      <span>Deepfake detected - Review evidence</span>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer Stats */}
        {history.length > 0 && (
          <div className="p-3 bg-muted/30 border-t border-border/50">
            <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground">
              <div className="flex items-center gap-4">
                <span>TOTAL: {history.length}</span>
                <span className="text-red-500">
                  FAKE: {history.filter(r => r.verdict === 'FAKE').length}
                </span>
                <span className="text-green-500">
                  REAL: {history.filter(r => r.verdict === 'REAL').length}
                </span>
              </div>
              {activeJobs && activeJobs.size > 0 && (
                <span className="text-blue-500">
                  {activeJobs.size} JOB{activeJobs.size > 1 ? 'S' : ''} RUNNING
                </span>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}