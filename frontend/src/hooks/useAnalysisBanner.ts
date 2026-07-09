import { useWebSocket } from '@/context/WebSocketContext';
import { ActiveJob } from '@/api';

export type AnalysisBannerVariant = 'processing' | 'completed' | 'active-elsewhere' | null;

interface AnalysisBannerState {
  variant: AnalysisBannerVariant;
  jobId: string | null;
  progress: number;
  stage: string;
  completedJobId: string | null;
  activeJobs: ActiveJob[];
}

/**
 * Resolves the single "active analysis" banner state from WebSocketContext,
 * shared between the visual banner and layout spacing that reserves room for it.
 */
export const useAnalysisBanner = (): AnalysisBannerState => {
  const { analysisState, completedJobId, activeJobs } = useWebSocket();
  const { isProcessing, progress, stage, jobId } = analysisState;

  let variant: AnalysisBannerVariant = null;
  if (isProcessing && jobId) {
    variant = 'processing';
  } else if (completedJobId) {
    variant = 'completed';
  } else if (activeJobs.length > 0) {
    variant = 'active-elsewhere';
  }

  return { variant, jobId, progress, stage, completedJobId, activeJobs };
};
