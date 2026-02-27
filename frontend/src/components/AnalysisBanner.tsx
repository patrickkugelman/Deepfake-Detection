import { useNavigate } from "react-router-dom";
import { useWebSocket } from "@/context/WebSocketContext";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle2, X } from "lucide-react";

/**
 * Persistent banner during analysis - allows navigation while job processes
 */
export function AnalysisBanner() {
  const navigate = useNavigate();
  const { analysisState, completedJobId, clearCompletedJob } = useWebSocket();
  const { isProcessing, progress, stage, jobId } = analysisState;

  if (!jobId) return null;

  // Analysis in progress
  if (isProcessing) {
    return (
      <div className="fixed top-16 left-0 right-0 z-40 bg-primary/90 text-primary-foreground px-4 py-2 flex items-center justify-between gap-4 shadow-lg animate-in slide-in-from-top-2">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="font-medium">Analiză în curs (Job: {jobId.slice(0, 8)}...)</span>
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

  // Analysis completed - show success banner
  if (completedJobId) {
    return (
      <div className="fixed top-16 left-0 right-0 z-40 bg-green-600 text-white px-4 py-2 flex items-center justify-between gap-4 shadow-lg animate-in slide-in-from-top-2">
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

  return null;
}
