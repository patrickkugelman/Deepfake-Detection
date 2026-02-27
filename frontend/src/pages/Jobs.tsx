import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { MainLayout } from '@/layouts/MainLayout';
import { PageHeader } from '@/components/shared/PageHeader';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import {
  getMyActiveJobs, ActiveJob, queryUserHistory, HistoryRecord, analyzeFile, api
} from '@/api';
import { useAuth } from '@/context/AuthContext';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';
import { toast } from 'sonner';
import {
  Clock, CheckCircle, XCircle, Loader2, RefreshCw, FileText, ExternalLink,
  Upload, X, Play, Eye, Download, FileVideo, ListOrdered, Trash2
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface QueueItem {
  id: string;           // local uuid
  file: File;
  status: 'waiting' | 'uploading' | 'processing' | 'done' | 'error';
  job_id?: string;
  progress: number;
  error?: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const verdictStyles: Record<string, string> = {
  REAL: 'bg-green-500/10 text-green-400 border-green-500/30',
  FAKE: 'bg-red-500/10 text-red-400 border-red-500/30',
  INCONCLUSIVE: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  ERROR: 'bg-muted text-muted-foreground border-muted-foreground/30',
  PENDING: 'bg-primary/10 text-primary border-primary/30',
};
const verdictLabels: Record<string, string> = {
  REAL: '✅ Autentic',
  FAKE: '⚠️ Deepfake',
  INCONCLUSIVE: '❓ Inconclusiv',
  ERROR: '❌ Eroare',
  PENDING: '⏳ În Procesare',
};

function StatusBadge({ status, progress }: { status: string; progress: number }) {
  switch (status) {
    case 'processing': return (
      <span className="flex items-center gap-1.5 text-blue-400 text-xs font-medium">
        <Loader2 className="h-3 w-3 animate-spin" /> Procesare {progress}%
      </span>
    );
    case 'queued': return (
      <span className="flex items-center gap-1.5 text-yellow-400 text-xs font-medium">
        <Clock className="h-3 w-3" /> În așteptare
      </span>
    );
    case 'completed': return (
      <span className="flex items-center gap-1.5 text-green-400 text-xs font-medium">
        <CheckCircle className="h-3 w-3" /> Finalizat
      </span>
    );
    case 'failed': return (
      <span className="flex items-center gap-1.5 text-red-400 text-xs font-medium">
        <XCircle className="h-3 w-3" /> Eșuat
      </span>
    );
    default: return <span className="text-gray-400 text-xs">{status}</span>;
  }
}

function formatDate(ts?: string | number | null) {
  if (!ts) return 'N/A';
  const date = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return isNaN(date.getTime()) ? 'N/A' : format(date, 'dd MMM yyyy, HH:mm');
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function Jobs() {
  const navigate = useNavigate();
  const { token } = useAuth();

  const [activeTab, setActiveTab] = useState<'active' | 'history' | 'queue'>('active');

  // Active jobs (polling)
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);
  const [activeLoading, setActiveLoading] = useState(true);

  // History
  const [historyJobs, setHistoryJobs] = useState<HistoryRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState(false);

  // Upload Queue
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const processingRef = useRef(false);
  const prevActiveJobsCountRef = useRef(0);

  // ── Fetch helpers ────────────────────────────────────────────────────────────

  const fetchActiveJobs = useCallback(async () => {
    try {
      const jobs = await getMyActiveJobs();
      setActiveJobs(jobs);
    } catch (err) {
      console.error('Error fetching active jobs:', err);
    } finally {
      setActiveLoading(false);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    if (!token) return;
    setHistoryLoading(true);
    setHistoryError(false);
    try {
      const data = await queryUserHistory(token, 100);
      setHistoryJobs(data.records || []);
    } catch (err) {
      console.error('Error fetching history:', err);
      setHistoryError(true);
    } finally {
      setHistoryLoading(false);
    }
  }, [token]);

  // Initial load + polling for active jobs
  useEffect(() => {
    fetchActiveJobs();
    fetchHistory();

    const activeInterval = setInterval(fetchActiveJobs, 3000);
    const historyInterval = setInterval(fetchHistory, 10000);
    return () => {
      clearInterval(activeInterval);
      clearInterval(historyInterval);
    };
  }, [fetchActiveJobs, fetchHistory]);

  // ── Queue processing ──────────────────────────────────────────────────────────

  const processQueue = useCallback(async (currentQueue: QueueItem[]) => {
    if (processingRef.current || !token) return;

    const nextItem = currentQueue.find(q => q.status === 'waiting');
    if (!nextItem) {
      setIsRunning(false);
      processingRef.current = false;
      return;
    }

    processingRef.current = true;
    setIsRunning(true);

    // Mark as uploading
    setQueue(prev => prev.map(q =>
      q.id === nextItem.id ? { ...q, status: 'uploading' as const, progress: 5 } : q
    ));

    try {
      const jobResponse = await analyzeFile(nextItem.file, token);

      // Mark as processing
      setQueue(prev => prev.map(q =>
        q.id === nextItem.id ? { ...q, status: 'processing' as const, job_id: jobResponse.job_id, progress: 15 } : q
      ));

      toast.info(`Analiza a început pentru ${nextItem.file.name}`);

      // Poll until done
      const poll = setInterval(async () => {
        try {
          const statusResp = await api.get<{ status: string; progress: number }>(`/jobs/${jobResponse.job_id}`);
          const { status, progress } = statusResp.data;

          setQueue(prev => prev.map(q =>
            q.id === nextItem.id ? { ...q, progress: progress || q.progress } : q
          ));

          if (status === 'completed') {
            clearInterval(poll);
            setQueue(prev => prev.map(q =>
              q.id === nextItem.id ? { ...q, status: 'done' as const, progress: 100 } : q
            ));
            toast.success(`✅ ${nextItem.file.name} — analiză finalizată!`, {
              action: { label: 'Vezi rezultat', onClick: () => navigate(`/results/${jobResponse.job_id}`) }
            });
            // Refresh history and process next item
            fetchHistory();
            processingRef.current = false;
            setQueue(prev => {
              processQueue(prev);
              return prev;
            });
          } else if (status === 'failed') {
            clearInterval(poll);
            setQueue(prev => prev.map(q =>
              q.id === nextItem.id ? { ...q, status: 'error' as const, error: 'Analiза a eșuat pe server' } : q
            ));
            toast.error(`❌ ${nextItem.file.name} — analiza a eșuat`);
            processingRef.current = false;
            setQueue(prev => {
              processQueue(prev);
              return prev;
            });
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 3000);

    } catch (err: any) {
      setQueue(prev => prev.map(q =>
        q.id === nextItem.id ? { ...q, status: 'error' as const, error: err.message || 'Eroare upload' } : q
      ));
      toast.error(`❌ ${nextItem.file.name} — eroare la upload`);
      processingRef.current = false;
      setQueue(prev => {
        processQueue(prev);
        return prev;
      });
    }
  }, [token, navigate, fetchHistory]);

  // ── Auto-start queue when active analysis finishes ───────────────────────────
  useEffect(() => {
    const wasActive = prevActiveJobsCountRef.current > 0;
    const isNowIdle = activeJobs.length === 0;
    const hasWaiting = queue.some(q => q.status === 'waiting');

    if (wasActive && isNowIdle && hasWaiting && !processingRef.current) {
      toast.info('✅ Analiza anterioară terminată — pornesc coada automat');
      processQueue(queue);
    }

    prevActiveJobsCountRef.current = activeJobs.length;
  }, [activeJobs, queue, processQueue]);

  const addFilesToQueue = (files: File[]) => {
    const allowed = ['video/', 'image/', 'audio/'];
    const valid = files.filter(f => allowed.some(t => f.type.startsWith(t)));
    if (valid.length < files.length) {
      toast.warning('Unele fișiere au fost ignorate (tip nesuportat)');
    }
    const newItems: QueueItem[] = valid.map(f => ({
      id: Math.random().toString(36).slice(2),
      file: f,
      status: 'waiting',
      progress: 0,
    }));
    setQueue(prev => [...prev, ...newItems]);
    setActiveTab('queue');
  };

  const removeFromQueue = (id: string) => {
    setQueue(prev => prev.filter(q => q.id !== id || q.status === 'processing' || q.status === 'uploading'));
  };

  const clearDone = () => {
    setQueue(prev => prev.filter(q => q.status === 'waiting' || q.status === 'uploading' || q.status === 'processing'));
  };

  const startQueue = () => {
    if (!processingRef.current) {
      processQueue(queue);
    }
  };

  // ── CSV Export ────────────────────────────────────────────────────────────────

  const exportCSV = () => {
    if (!historyJobs.length) return;
    const csv = ['ID,Filename,Verdict,Score,Data'].concat(
      historyJobs.map((a: HistoryRecord) =>
        `${a.job_id || a.id},${a.filename},${a.verdict},${((a.confidence || 0) * 100).toFixed(1)},${a.timestamp || a.created_at || ''}`
      )
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'deepguard_istoric.csv';
    link.click();
  };

  // ── Queue status icon ─────────────────────────────────────────────────────────

  const queueWaiting = queue.filter(q => q.status === 'waiting').length;
  const queueProcessing = queue.filter(q => q.status === 'processing' || q.status === 'uploading').length;
  const queueDone = queue.filter(q => q.status === 'done').length;
  const queueError = queue.filter(q => q.status === 'error').length;

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <MainLayout>
      <div className="p-6 lg:p-8 max-w-6xl mx-auto">
        <PageHeader
          title="Analizele Mele"
          description="Urmărește analizele active, istoricul complet și procesează fișiere în serie"
          actions={
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => { fetchActiveJobs(); fetchHistory(); }}>
                <RefreshCw className="h-4 w-4 mr-2" /> Reîncarcă
              </Button>
              <label>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  className="hidden"
                  accept="video/*,image/*,audio/*"
                  onChange={e => {
                    if (e.target.files) addFilesToQueue(Array.from(e.target.files));
                    e.target.value = '';
                  }}
                />
                <Button size="sm" onClick={() => fileInputRef.current?.click()}>
                  <Upload className="h-4 w-4 mr-2" /> Adaugă Fișiere
                </Button>
              </label>
            </div>
          }
        />

        {/* ── Tabs ─────────────────────────────────────────────────────────── */}
        <div className="flex gap-1 mb-6 bg-muted/30 rounded-lg p-1 w-fit">
          <button
            onClick={() => setActiveTab('active')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'active' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
          >
            ⏳ Active
            {activeJobs.length > 0 && (
              <span className="ml-1.5 bg-primary text-primary-foreground text-xs px-1.5 py-0.5 rounded-full">
                {activeJobs.length}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'history' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
          >
            📋 Istoric
            {historyJobs.length > 0 && (
              <span className="ml-1.5 text-xs text-muted-foreground">({historyJobs.length})</span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('queue')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === 'queue' ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
          >
            <span className="flex items-center gap-1.5">
              <ListOrdered className="h-3.5 w-3.5" /> Coadă Upload
              {queue.length > 0 && (
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${isRunning ? 'bg-blue-500 text-white' : 'bg-muted text-muted-foreground'}`}>
                  {queue.length}
                </span>
              )}
            </span>
          </button>
        </div>

        {/* ── Tab: Active Jobs ──────────────────────────────────────────────── */}
        {activeTab === 'active' && (
          <div>
            {activeLoading ? (
              <div className="space-y-4">
                {[1, 2].map(i => <Skeleton key={i} className="h-24 w-full rounded-xl" />)}
              </div>
            ) : activeJobs.length === 0 ? (
              <div className="text-center py-20 text-muted-foreground glass-card rounded-xl">
                <CheckCircle className="h-14 w-14 mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium">Nicio analiză activă</p>
                <p className="text-sm mt-1">Toate analizele sunt finalizate</p>
                <Button className="mt-5" onClick={() => navigate('/upload')}>
                  <Play className="h-4 w-4 mr-2 fill-current" /> Analiză Nouă
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {activeJobs.map(job => (
                  <div key={job.job_id} className="border border-border rounded-xl p-5 bg-card">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
                        <div>
                          <p className="font-medium text-foreground">{job.filename || 'Fișier necunoscut'}</p>
                          <p className="text-xs text-muted-foreground mt-0.5">{job.status_message || 'Se procesează...'}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <StatusBadge status={job.status} progress={job.progress} />
                        <Button variant="ghost" size="sm" onClick={() => navigate(`/results/${job.job_id}`)}>
                          <ExternalLink className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </div>
                    <Progress value={job.progress || 0} className="h-2" />
                    <p className="text-xs text-muted-foreground mt-1 text-right">{job.progress || 0}%</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Tab: History ─────────────────────────────────────────────────── */}
        {activeTab === 'history' && (
          <div>
            <div className="flex justify-end mb-3 gap-2">
              <Button variant="outline" size="sm" onClick={() => fetchHistory()}>
                <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Actualizează
              </Button>
              <Button variant="outline" size="sm" onClick={exportCSV} disabled={!historyJobs.length}>
                <Download className="h-3.5 w-3.5 mr-1.5" /> Export CSV
              </Button>
            </div>

            <div className="glass-card rounded-xl overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Fișier</TableHead>
                    <TableHead>Verdict</TableHead>
                    <TableHead>Scor</TableHead>
                    <TableHead>Data</TableHead>
                    <TableHead className="text-right">Acțiuni</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {historyLoading ? (
                    Array.from({ length: 5 }).map((_, i) => (
                      <TableRow key={i}>
                        <TableCell><Skeleton className="h-5 w-36" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-12" /></TableCell>
                        <TableCell><Skeleton className="h-5 w-28" /></TableCell>
                        <TableCell><Skeleton className="h-8 w-8 rounded ml-auto" /></TableCell>
                      </TableRow>
                    ))
                  ) : historyError ? (
                    <TableRow>
                      <TableCell colSpan={5} className="text-center py-10 text-destructive">
                        Eroare la încărcarea istoricului. Verifică conexiunea cu backend-ul.
                      </TableCell>
                    </TableRow>
                  ) : historyJobs.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} className="text-center py-12 text-muted-foreground">
                        <FileText className="h-10 w-10 mx-auto mb-3 opacity-20" />
                        Nu există analize anterioare.
                      </TableCell>
                    </TableRow>
                  ) : (
                    historyJobs.map(a => (
                      <TableRow
                        key={a.job_id || a.id || a.file_hash || Math.random().toString()}
                        className="cursor-pointer hover:bg-muted/50 transition-colors"
                        onClick={() => navigate(`/results/${a.job_id || a.file_hash || a.id}`)}
                      >
                        <TableCell>
                          <div className="flex items-center gap-2 font-medium max-w-[220px]">
                            <FileVideo className="h-4 w-4 text-primary/70 shrink-0" />
                            <span className="truncate">{a.filename}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={cn('w-fit', verdictStyles[a.verdict] || verdictStyles.INCONCLUSIVE)}
                          >
                            {a.verdict === 'PENDING' && <RefreshCw className="h-3 w-3 mr-1 animate-spin" />}
                            {verdictLabels[a.verdict] || a.verdict}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {a.verdict === 'PENDING' ? (
                            <span className="text-muted-foreground text-xs italic">Se calculează...</span>
                          ) : (
                            <span className={cn(
                              'font-bold text-sm',
                              a.verdict === 'FAKE' ? 'text-red-400' :
                                a.verdict === 'REAL' ? 'text-green-400' : 'text-yellow-400'
                            )}>
                              {((a.confidence || 0) * 100).toFixed(0)}%
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="text-muted-foreground text-sm">
                          {formatDate(a.timestamp || a.created_at)}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button variant="ghost" size="icon" onClick={e => {
                            e.stopPropagation();
                            navigate(`/results/${a.job_id || a.file_hash || a.id}`);
                          }}>
                            <Eye className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        )}

        {/* ── Tab: Upload Queue ────────────────────────────────────────────── */}
        {activeTab === 'queue' && (
          <div className="space-y-4">
            {/* Drop zone */}
            <div
              className="border-2 border-dashed border-border rounded-xl p-10 text-center hover:border-primary/40 transition-colors cursor-pointer bg-card/30"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault();
                addFilesToQueue(Array.from(e.dataTransfer.files));
              }}
            >
              <Upload className="h-10 w-10 mx-auto mb-3 text-muted-foreground" />
              <p className="font-medium">Trage fișiere sau click pentru a adăuga în coadă</p>
              <p className="text-xs text-muted-foreground mt-1">Video, Imagine, Audio — multiple fișiere simultan</p>
            </div>

            {/* Queue controls */}
            {queue.length > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex gap-3 text-xs text-muted-foreground">
                  {queueWaiting > 0 && <span className="flex items-center gap-1"><Clock className="h-3 w-3 text-yellow-400" /> {queueWaiting} în așteptare</span>}
                  {queueProcessing > 0 && <span className="flex items-center gap-1"><Loader2 className="h-3 w-3 animate-spin text-blue-400" /> {queueProcessing} procesare</span>}
                  {queueDone > 0 && <span className="flex items-center gap-1"><CheckCircle className="h-3 w-3 text-green-400" /> {queueDone} finalizat</span>}
                  {queueError > 0 && <span className="flex items-center gap-1"><XCircle className="h-3 w-3 text-red-400" /> {queueError} eroare</span>}
                </div>
                <div className="flex gap-2">
                  {queueDone + queueError > 0 && (
                    <Button variant="ghost" size="sm" onClick={clearDone}>
                      <Trash2 className="h-3.5 w-3.5 mr-1.5" /> Șterge finalizate
                    </Button>
                  )}
                  {queueWaiting > 0 && !isRunning && (
                    activeJobs.length > 0 ? (
                      <div title="Se va porni automat când analiza curentă se termină">
                        <Button size="sm" disabled>
                          <Clock className="h-3.5 w-3.5 mr-1.5" /> Pornire automată...
                        </Button>
                      </div>
                    ) : (
                      <Button size="sm" onClick={startQueue}>
                        <Play className="h-3.5 w-3.5 mr-1.5 fill-current" /> Pornește Coada
                      </Button>
                    )
                  )}
                  {isRunning && (
                    <Button size="sm" variant="outline" disabled>
                      <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" /> Se procesează...
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* Queue items */}
            {queue.length === 0 ? (
              <div className="text-center py-10 text-muted-foreground">
                <ListOrdered className="h-10 w-10 mx-auto mb-3 opacity-20" />
                <p>Coada este goală</p>
                <p className="text-xs mt-1">Adaugă fișiere pentru a procesa mai multe analize în serie</p>
              </div>
            ) : (
              <div className="space-y-2">
                {queue.map((item, idx) => (
                  <div key={item.id} className={cn(
                    'border rounded-xl p-4 bg-card transition-all',
                    item.status === 'done' ? 'border-green-500/20 bg-green-500/5' :
                      item.status === 'error' ? 'border-red-500/20 bg-red-500/5' :
                        item.status === 'processing' || item.status === 'uploading' ? 'border-primary/30 bg-primary/5' :
                          'border-border'
                  )}>
                    <div className="flex items-center gap-3">
                      {/* Order number */}
                      <span className="text-xs text-muted-foreground font-mono w-5 text-center shrink-0">
                        {idx + 1}
                      </span>

                      {/* Status icon */}
                      <div className="shrink-0">
                        {item.status === 'waiting' && <Clock className="h-4 w-4 text-yellow-400" />}
                        {(item.status === 'uploading' || item.status === 'processing') && <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />}
                        {item.status === 'done' && <CheckCircle className="h-4 w-4 text-green-400" />}
                        {item.status === 'error' && <XCircle className="h-4 w-4 text-red-400" />}
                      </div>

                      {/* File info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2">
                          <p className="font-medium text-sm truncate">{item.file.name}</p>
                          <span className="text-xs text-muted-foreground shrink-0">
                            {(item.file.size / 1024 / 1024).toFixed(1)} MB
                          </span>
                        </div>

                        {/* Progress bar for active items */}
                        {(item.status === 'uploading' || item.status === 'processing') && (
                          <Progress value={item.progress} className="h-1.5 mt-2" />
                        )}

                        {item.status === 'error' && item.error && (
                          <p className="text-xs text-red-400 mt-1">{item.error}</p>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-1 shrink-0">
                        {item.status === 'done' && item.job_id && (
                          <Button variant="ghost" size="sm" onClick={() => navigate(`/results/${item.job_id}`)}>
                            <Eye className="h-3.5 w-3.5 mr-1" /> Rezultat
                          </Button>
                        )}
                        {item.status === 'waiting' && (
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-red-400" onClick={() => removeFromQueue(item.id)}>
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </MainLayout>
  );
}
