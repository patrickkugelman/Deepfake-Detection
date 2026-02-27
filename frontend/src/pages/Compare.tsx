import { MainLayout } from '@/layouts/MainLayout';
import { PageHeader } from '@/components/shared/PageHeader';
import { queryUserHistory, HistoryRecord } from '@/api';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { useAuth } from '@/context/AuthContext';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';

const formatTimestamp = (ts: unknown): string => {
  if (!ts) return "N/A";
  const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts as string);
  return isNaN(date.getTime()) ? "N/A" : format(date, 'yyyy-MM-dd HH:mm:ss');
};

export default function Compare() {
  const { token } = useAuth();

  // Fetch real history
  const { data: historyData } = useQuery({
    queryKey: ['history', token],
    queryFn: () => queryUserHistory(token || "", 100),
    enabled: !!token
  });

  const records = historyData?.records || [];

  const [leftId, setLeftId] = useState<string>("");
  const [rightId, setRightId] = useState<string>("");

  const leftAnalysis = records.find(r => (r.job_id || r.id) === leftId);
  const rightAnalysis = records.find(r => (r.job_id || r.id) === rightId);

  const verdictStyles: Record<string, string> = {
    REAL: 'text-success',
    FAKE: 'text-destructive',
    INCONCLUSIVE: 'text-warning',
    ERROR: 'text-muted-foreground'
  };

  return (
    <MainLayout>
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <PageHeader title="Comparație Side-by-Side" description="Compară rezultatele a două analize din istoricul tău" />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { value: leftId, set: setLeftId, analysis: leftAnalysis, label: "Fișier A" },
            { value: rightId, set: setRightId, analysis: rightAnalysis, label: "Fișier B" }
          ].map((side, i) => (
            <div key={i} className="space-y-4">
              <div className="flex flex-col gap-2">
                <label className="text-sm font-medium text-muted-foreground">{side.label}</label>
                <Select value={side.value} onValueChange={side.set}>
                  <SelectTrigger><SelectValue placeholder="Selectează o analiză..." /></SelectTrigger>
                  <SelectContent>
                    {records.map(a => (
                      <SelectItem key={a.job_id || a.id || ''} value={a.job_id || a.id || ''}>
                        {a.filename} <span className="text-xs text-muted-foreground ml-2">({formatTimestamp(a.timestamp || a.created_at).split(' ')[0]})</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {side.analysis ? (
                <div className="glass-card rounded-xl p-6 space-y-6 animate-in fade-in slide-in-from-bottom-4">
                  <div className="text-center border-b border-border pb-4">
                    <h3 className="text-xl font-semibold mb-1 truncate" title={side.analysis.filename}>{side.analysis.filename}</h3>
                    <p className="text-sm text-muted-foreground">{formatTimestamp(side.analysis.timestamp || side.analysis.created_at)}</p>
                  </div>

                  <div className="text-center py-4">
                    <p className={cn("text-3xl font-bold tracking-wider", verdictStyles[side.analysis.verdict || 'INCONCLUSIVE'])}>
                      {side.analysis.verdict}
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">VERDICT FINAL</p>
                  </div>

                  {/* Mock breakdown until we pull full details */}
                  <div className="space-y-3 bg-muted/30 p-4 rounded-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Scor Încredere</span>
                      <Badge variant="outline">{((side.analysis.confidence ?? 0) * 100).toFixed(0)}%</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Hash</span>
                      <span className="text-xs font-mono text-muted-foreground truncate max-w-[100px]">{(side.analysis.file_hash || side.analysis.job_id || 'N/A').toString().substring(0, 8)}...</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-64 border-2 border-dashed border-border rounded-xl flex items-center justify-center text-muted-foreground">
                  Selectează un fișier pentru a vedea detaliile
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </MainLayout>
  );
}
