import { MainLayout } from '@/layouts/MainLayout';
import { PageHeader } from '@/components/shared/PageHeader';
import { Button } from '@/components/ui/button';
import { Upload as UploadIcon, FileVideo, X, Play, Loader2, Terminal, ListOrdered } from 'lucide-react';
import { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Progress } from '@/components/ui/progress';
import { analyzeFile, getJobStatus } from '@/api';
import { useAuth } from '@/context/AuthContext';
import { toast } from 'sonner';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ScrollArea } from '@/components/ui/scroll-area';

export default function Upload() {
  const navigate = useNavigate();
  const { token } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [forceRefresh, setForceRefresh] = useState(false);

  // Consumăm starea globală a analizei din WebSocketContext
  const { analysisState, setAnalysisState, resetAnalysis, isConnected } = useWebSocket();
  const { isProcessing, progress, stage, logs, jobId } = analysisState;

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 1. Redirecționare automată la finalizare
  // 1. Redirecționare automată (Removed - we redirect immediately now)
  // useEffect(() => {
  //   if (!isProcessing && progress === 100 && jobId) {
  //     toast.success("Analiză Completă!");
  //     const timer = setTimeout(() => {
  //       navigate(`/results/${jobId}`);
  //     }, 1500);
  //     return () => clearTimeout(timer);
  //   }
  // }, [isProcessing, progress, jobId, navigate]);

  // Curățăm intervalul de polling la demontarea componentei
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) setFile(droppedFile);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) setFile(selectedFile);
  };

  // 2. Backup Polling (în caz că WebSocket-ul are latență)
  const startPolling = useCallback((targetId: string) => {
    if (!token) return;
    // Clear existing to avoid duplicates
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

    pollIntervalRef.current = setInterval(async () => {
      try {
        const stats = await getJobStatus(targetId, token);
        if (stats.status === 'completed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setAnalysisState(prev => ({ ...prev, progress: 100, isProcessing: false }));
        } else if (stats.status === 'failed') {
          if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          setAnalysisState(prev => ({ ...prev, isProcessing: false, stage: 'Eroare la procesare' }));
          toast.error("Analiza a eșuat pe server.");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 3000);
  }, [token, setAnalysisState]);

  // Resume polling on mount if active
  useEffect(() => {
    if (isProcessing && jobId) {
      startPolling(jobId);
    }
  }, [startPolling, isProcessing, jobId]);

  // 3. Validare + Declanșarea Analizei
  const FILE_SIZE_LIMITS: Record<string, number> = {
    'video': 500 * 1024 * 1024,   // 500MB
    'image': 50 * 1024 * 1024,    // 50MB
    'audio': 100 * 1024 * 1024,   // 100MB
  };

  const handleAnalysis = async () => {
    if (!file || !token) return;

    // Frontend validation
    const fileType = file.type.split('/')[0]; // video, image, audio
    const maxSize = FILE_SIZE_LIMITS[fileType] || 500 * 1024 * 1024;

    if (file.size > maxSize) {
      const maxMB = Math.round(maxSize / 1024 / 1024);
      toast.error(`Fișier prea mare`, { description: `Limita pentru ${fileType} este ${maxMB}MB. Fișierul tău are ${(file.size / 1024 / 1024).toFixed(1)}MB.` });
      return;
    }

    const allowedTypes = ['video/', 'image/', 'audio/'];
    if (!allowedTypes.some(t => file.type.startsWith(t))) {
      toast.error('Tip de fișier neacceptat', { description: 'Acceptăm doar video, imagini și audio.' });
      return;
    }

    resetAnalysis(); // Resetăm starea anterioară
    setAnalysisState(prev => ({
      ...prev,
      isProcessing: true,
      stage: 'Se inițializează Cortex...',
      progress: 5
    }));

    try {
      const jobResponse = await analyzeFile(file, token, undefined, forceRefresh);

      setAnalysisState(prev => ({
        ...prev,
        jobId: jobResponse.job_id,
        stage: 'Fișier acceptat. În curs de distribuire către agenți...',
        progress: 15
      }));

      toast.info("Analiza a început", { description: "Cortex coordonează agenții de detecție." });

      // Navigate immediately - no blocking
      navigate(`/results/${jobResponse.job_id}`);

    } catch (error: any) {
      setAnalysisState(prev => ({ ...prev, isProcessing: false, progress: 0 }));
      toast.error('Eroare la pornirea analizei', { description: error.message });
    }
  };

  return (
    <MainLayout>
      <div className="p-6 lg:p-8 max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
        <PageHeader
          title="Analiză Nouă Forensic"
          description="Încărcați probe media pentru verificare multi-agent prin Cortex AI."
        />

        {/* Zona de Drop */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={cn(
            "border-2 border-dashed rounded-3xl p-16 text-center transition-all duration-300 relative overflow-hidden",
            isDragging ? "border-primary bg-primary/5 scale-[1.01]" : "border-slate-800 hover:border-primary/40 bg-card/30",
            file && "border-green-500/30 bg-green-500/5",
            isProcessing && "opacity-50 pointer-events-none"
          )}
        >
          {!file ? (
            <>
              <div className="bg-primary/10 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <UploadIcon className="h-10 w-10 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2">Trageți proba video, imagine sau audio aici</h3>
              <p className="text-muted-foreground text-sm mb-6 uppercase tracking-widest font-medium">sau</p>
              <label className="cursor-pointer">
                <input type="file" onChange={handleFileChange} className="hidden" accept="video/*,image/*,audio/*,.mp4,.avi,.mov,.mkv,.webm,.jpg,.jpeg,.png,.webp,.gif,.mp3,.wav,.flac" />
                <div className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-secondary text-secondary-foreground hover:bg-secondary/80 h-10 px-4 py-2">
                  <span className="px-8">Alege din Explorer</span>
                </div>
              </label>
            </>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <div className="bg-green-500/20 p-4 rounded-full">
                <FileVideo className="h-12 w-12 text-green-500" />
              </div>
              <div className="space-y-1">
                <p className="font-bold text-lg">{file.name}</p>
                <p className="text-xs text-muted-foreground uppercase">{(file.size / 1024 / 1024).toFixed(2)} MB • READY FOR CORTEX</p>
              </div>
              {!isProcessing && (
                <Button variant="ghost" size="sm" onClick={() => setFile(null)} className="text-red-400 hover:text-red-300 hover:bg-red-400/10">
                  Elimină fișierul
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Opțiuni Suplimentare */}
        {!isProcessing && (
          <div className="flex flex-col items-center gap-6">
            <div className="flex items-center space-x-3 bg-slate-900/50 px-6 py-3 rounded-full border border-white/5 shadow-inner">
              <Checkbox id="force-refresh" checked={forceRefresh} onCheckedChange={(c) => setForceRefresh(!!c)} />
              <Label htmlFor="force-refresh" className="text-xs font-bold uppercase tracking-wider cursor-pointer text-slate-400">
                Ignoră Memoria Cache (Deep Scan)
              </Label>
            </div>

            {file && (
              <div className="w-full max-w-md flex flex-col gap-3">
                <Button onClick={handleAnalysis} className="w-full h-14 text-lg font-bold gap-3 shadow-xl shadow-primary/20" size="lg">
                  <Play className="h-6 w-6 fill-current" /> INIȚIAZĂ EXPERTIZA
                </Button>
                <Button
                  variant="outline"
                  className="w-full gap-2 text-sm"
                  onClick={() => navigate('/jobs', { state: { addToQueue: true } })}
                >
                  <ListOrdered className="h-4 w-4" /> Adaugă în Coadă (Analizele Mele)
                </Button>
              </div>
            )}
          </div>
        )}

        {/* Vizualizarea Procesului (Progress & Logs) */}
        {isProcessing && (
          <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
            <div className="glass-card rounded-2xl p-8 border border-primary/20 bg-primary/5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  <span className="font-bold text-sm uppercase tracking-tighter">{stage}</span>
                </div>
                <span className="font-mono text-xl font-black">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-3" />
            </div>

            {/* Consola Live de Loguri */}
            <div className="bg-black/80 rounded-2xl p-6 border border-white/5 font-mono text-xs shadow-2xl relative">
              <div className="absolute top-4 right-6 flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-[10px] text-slate-500 font-bold uppercase">{isConnected ? 'Link Active' : 'Link Offline'}</span>
              </div>

              <div className="flex items-center gap-2 mb-4 border-b border-white/10 pb-3">
                <Terminal className="h-4 w-4 text-primary" />
                <span className="text-primary font-bold">CORTEX_STREAM_OUTPUT</span>
              </div>

              <ScrollArea className="h-48 w-full pr-4">
                <div className="space-y-1.5">
                  {logs.map((log: any, i: number) => (
                    <div key={i} className="flex gap-3 items-start animate-in fade-in slide-in-from-left-1">
                      <span className="text-slate-600 shrink-0 font-bold">{'>'}</span>
                      <span className={cn(
                        log.status === 'success' ? 'text-green-400' :
                          log.status === 'thinking' ? 'text-blue-400' :
                            'text-slate-400'
                      )}>
                        {log.message}
                      </span>
                    </div>
                  ))}
                  <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
                </div>
              </ScrollArea>
            </div>
          </div>
        )}
      </div>
    </MainLayout>
  );
}