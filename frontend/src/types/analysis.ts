export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type Verdict = 'authentic' | 'deepfake' | 'inconclusive';
export type AgentType = 'vision' | 'audio' | 'bio';

export interface AgentResult {
  agent: AgentType;
  score: number; // 0-100, higher = more authentic
  confidence: number; // 0-100
  status: AnalysisStatus;
  details?: string;
  anomalies?: string[];
}

export interface SymbolicReasoning {
  conclusion: string;
  rules: ReasoningRule[];
  overriddenPredictions: OverriddenPrediction[];
  logicScore: number; // 0-100
}

export interface ReasoningRule {
  id: string;
  description: string;
  condition: string;
  result: 'pass' | 'fail' | 'warning';
  weight: number;
}

export interface OverriddenPrediction {
  agent: AgentType;
  originalScore: number;
  overriddenTo: number;
  reason: string;
}

export interface FrameAnomaly {
  frameNumber: number;
  timestamp: number;
  severity: 'low' | 'medium' | 'high';
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AudioAnomaly {
  timestamp: number;
  duration: number;
  type: string;
  severity: 'low' | 'medium' | 'high';
  frequency?: number;
}

export interface TimelineEvent {
  timestamp: number;
  type: 'visual' | 'audio' | 'bio';
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface Visualizations {
  frameAnomalies: FrameAnomaly[];
  audioSpectrum: number[];
  timeline: TimelineEvent[];
  heatmapData: number[][];
}

export interface Analysis {
  id: string;
  filename: string;
  fileSize: number;
  duration: number; // in seconds
  createdAt: Date;
  completedAt?: Date;
  status: AnalysisStatus;
  verdict?: Verdict;
  overallScore?: number; // 0-100
  agentResults: AgentResult[];
  symbolicReasoning?: SymbolicReasoning;
  visualizations?: Visualizations;
  thumbnailUrl?: string;
  processingTime?: number; // in ms
}

export interface AnalysisStats {
  totalAnalyses: number;
  deepfakesDetected: number;
  authenticVideos: number;
  inconclusiveResults: number;
  averageProcessingTime: number;
  todayAnalyses: number;
  weeklyTrend: number; // percentage change
}

export interface UploadProgress {
  stage: 'uploading' | 'vision' | 'audio' | 'bio' | 'reasoning' | 'complete';
  progress: number; // 0-100
  currentAgent?: AgentType;
  message: string;
}

export interface JobDetails {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  message: string;
  verdict?: "REAL" | "FAKE" | "SUSPICIOUS" | "INCONCLUSIVE";
  confidence?: number;
  reasoning?: string;

  // File metadata
  filename: string;
  file_size?: number;
  mime_type?: string;
  resolution?: string;

  // Timestamps
  created_at?: string;
  completed_at?: string;

  // Extra for compatibility if needed
  result?: any;
}
