import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useWebSocket } from "@/context/WebSocketContext";
import { useAnalysisBanner } from "@/hooks/useAnalysisBanner";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle2, X } from "lucide-react";

/**
 * Single global "active analysis" banner — resolves processing / completed /
 * active-elsewhere-on-account state via useAnalysisBanner (WebSocketContext
 * is the sole source of truth). Rendered once, globally, in App.tsx.
 */
export function AnalysisBanner() {
  const navigate = useNavigate();
  const { clearCompletedJob } = useWebSocket();
  const { variant, jobId, progress, stage, completedJobId, activeJobs } = useAnalysisBanner();
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (activeJobs.length === 0) setDismissed(false);
  }, [activeJobs.length]);

  if (variant === null) return null;
  if (variant === 'active-elsewhere' && dismissed) return null;

  if (variant === 'processing') {
    return (
      <div className="fixed top-0 left-0 right-0 z-40 bg-primary/90 text-primary-foreground px-4 py-2 flex items-center justify-between gap-4 shadow-lg animate-in slide-in-from-top-2">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="font-medium">Analiză în curs (Job: {jobId!.slice(0, 8)}...)</span>
          <span className="text-sm opacity-90">{stage}</span>
          <span className="text-sm font-mono">{Math.round(progress)}%</span>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => navigate(`/results/${jobId}`)}
          className="bg-white/20 hover:bg-white/30"
        >
          Vezi progres
        </Button>
      </div>
    );
  }

  if (variant === 'completed') {
    return (
      <div className="fixed top-0 left-0 right-0 z-40 bg-green-600 text-white px-4 py-2 flex items-center justify-between gap-4 shadow-lg animate-in slide-in-from-top-2">
        <div className="flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5" />
          <span className="font-medium">Analiză completă!</span>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              navigate(`/results/${completedJobId}`);
              clearCompletedJob();
            }}
            className="bg-white/20 hover:bg-white/30"
          >
            Vezi rezultate
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-white hover:bg-white/20"
            onClick={clearCompletedJob}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  // active-elsewhere: job(s) active on the account not tracked by this tab's WebSocketContext
  const job = activeJobs[0];
  const progressPct = job.progress || 0;

  return (
    <div className="fixed top-0 left-0 right-0 z-40 bg-primary text-primary-foreground px-4 py-2.5 flex items-center justify-between gap-4 shadow-lg">
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <Loader2 className="h-4 w-4 animate-spin shrink-0" />
        <span className="text-sm font-medium truncate">
          ⏳ Analiză în curs: <span className="font-bold">{job.filename || 'fișier'}</span>
          {job.status_message && <span className="opacity-80 ml-2">— {job.status_message}</span>}
        </span>
        <div className="hidden sm:flex items-center gap-2 shrink-0">
          <div className="w-24 bg-primary-foreground/20 rounded-full h-1.5">
            <div className="bg-primary-foreground h-1.5 rounded-full transition-all duration-500" style={{ width: `${progressPct}%` }} />
          </div>
          <span className="text-xs opacity-80">{progressPct}%</span>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <button
          onClick={() => navigate('/jobs')}
          className="text-xs underline underline-offset-2 opacity-90 hover:opacity-100 whitespace-nowrap"
        >
          Vezi progresul
        </button>
        <button onClick={() => setDismissed(true)} className="opacity-70 hover:opacity-100 ml-1">
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
