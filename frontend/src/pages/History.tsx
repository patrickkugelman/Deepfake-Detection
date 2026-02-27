import { MainLayout } from '@/layouts/MainLayout';
import { PageHeader } from '@/components/shared/PageHeader';
import { queryUserHistory, HistoryRecord } from '@/api';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Eye, Download, FileVideo, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { cn } from '@/lib/utils';
import { useQuery } from '@tanstack/react-query';
import { Skeleton } from '@/components/ui/skeleton';
import { useAuth } from '@/context/AuthContext';

export default function History() {
  const navigate = useNavigate();
  const { token } = useAuth();

  const verdictStyles: Record<string, string> = {
    REAL: 'bg-success/10 text-success border-success/30',
    FAKE: 'bg-destructive/10 text-destructive border-destructive/30',
    INCONCLUSIVE: 'bg-warning/10 text-warning border-warning/30',
    ERROR: 'bg-muted text-muted-foreground border-muted-foreground/30',
    PENDING: 'bg-primary/10 text-primary border-primary/30'
  };
  const verdictLabels: Record<string, string> = {
    REAL: 'Autentic',
    FAKE: 'Deepfake',
    INCONCLUSIVE: 'Inconclusiv',
    ERROR: 'Eroare',
    PENDING: 'În Procesare'
  };

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['history', token], // Re-fetch on token change
    queryFn: () => queryUserHistory(token || "", 50),
    enabled: !!token,
    refetchInterval: 10000 // Poll every 10s for updates
  });

  const historyRecords = data?.records || [];

  const exportCSV = () => {
    if (!historyRecords.length) return;
    const csv = ['ID,Filename,Verdict,Score,Date,Hash'].concat(
      historyRecords.map((a: HistoryRecord) => `${a.id},${a.filename},${a.verdict},${a.confidence?.toFixed(1)},${a.timestamp},${a.file_hash}`)
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'deepguard_analysis_history.csv'; a.click();
  };

  return (
    <MainLayout>
      <div className="p-6 lg:p-8 max-w-7xl mx-auto">
        <PageHeader title="Istoric Analize" description="Toate analizele efectuate (salvate in Neo4j)" actions={
          <div className="flex gap-2">
            <Button variant="outline" size="icon" onClick={() => refetch()}><RefreshCw className="h-4 w-4" /></Button>
            <Button variant="outline" onClick={exportCSV} disabled={!historyRecords.length}><Download className="h-4 w-4 mr-2" />Export CSV</Button>
          </div>
        } />
        <div className="glass-card rounded-xl overflow-hidden min-h-[400px]">
          <Table>
            <TableHeader>
              <TableRow><TableHead>Fișier</TableHead><TableHead>Verdict</TableHead><TableHead>Scor</TableHead><TableHead>Data</TableHead><TableHead></TableHead></TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell><Skeleton className="h-6 w-32" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-12" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-8 w-8 rounded-full" /></TableCell>
                  </TableRow>
                ))
              ) : isError ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-10 text-destructive">
                    Eroare la încărcarea istoricului. Verifică conexiunea cu backend-ul.
                  </TableCell>
                </TableRow>
              ) : historyRecords.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-10 text-muted-foreground">
                    Nu există analize anterioare.
                  </TableCell>
                </TableRow>
              ) : (
                historyRecords.map((a: HistoryRecord) => (
                  <TableRow key={a.job_id || a.id || a.file_hash || ''} className="cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => navigate(`/results/${a.job_id || a.file_hash || a.id}`)}>
                    <TableCell><div className="flex items-center gap-3 font-medium"><FileVideo className="h-5 w-5 text-primary/70" />{a.filename}</div></TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn("flex items-center gap-1 w-fit", verdictStyles[a.verdict] || verdictStyles.INCONCLUSIVE)}>
                        {a.verdict === 'PENDING' && <RefreshCw className="h-3 w-3 animate-spin" />}
                        {verdictLabels[a.verdict] || a.verdict}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {a.verdict === 'PENDING' ? (
                        <span className="text-muted-foreground text-xs italic">Se calculează...</span>
                      ) : (
                        <span className={cn(
                          "font-bold",
                          a.verdict === 'FAKE' ? 'text-destructive' :
                            a.verdict === 'REAL' ? 'text-success' : 'text-warning'
                        )}>
                          {(a.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {(() => {
                        const ts = a.timestamp || a.created_at;
                        if (!ts) return "N/A";
                        const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
                        return isNaN(date.getTime()) ? "Invalid date" : format(date, 'dd MMM yyyy, HH:mm');
                      })()}
                    </TableCell>
                    <TableCell><Button variant="ghost" size="icon"><Eye className="h-4 w-4" /></Button></TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </MainLayout>
  );
}
