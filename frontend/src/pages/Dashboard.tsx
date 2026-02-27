import { useEffect, useState, useRef } from "react";
import { useAuth } from "@/context/AuthContext";
import { 
  getDashboardStats, 
  DashboardStats, 
  getJobStatus, 
  queryUserHistory, 
  HistoryRecord, 
  Evidence,
  AnalysisResponse,
  createWebSocket
} from "@/api";
import { AgentPulse } from "@/components/dashboard/AgentPulse";
import { ForensicGallery } from "@/components/dashboard/ForensicGallery";
import { StatCard } from "@/components/dashboard/StatCard";
import { RecentAnalysisCard } from "@/components/dashboard/RecentAnalysisCard";
import { Shield, FileVideo, Activity, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { MainLayout } from "@/layouts/MainLayout";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

export default function Dashboard() {
  const { token, user } = useAuth();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentHistory, setRecentHistory] = useState<HistoryRecord[]>([]);
  const [artifacts, setArtifacts] = useState<Evidence[]>([]);
  const [synthIDScore, setSynthIDScore] = useState(0);
  const [activeJobs, setActiveJobs] = useState<Map<string, number>>(new Map());
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket Connection
  useEffect(() => {
    if (!token) return;

    const ws = createWebSocket((message) => {
      console.log('📡 WebSocket message:', message);
      
      if (message.type === 'status' && message.job_id) {
        setActiveJobs(prev => {
          const newMap = new Map(prev);
          
          if (message.status === 'completed' || message.status === 'failed') {
            newMap.delete(message.job_id);
            refreshHistory();
          } else {
            newMap.set(message.job_id, message.progress || 0);
          }
          
          return newMap;
        });
      }
    });

    wsRef.current = ws;
    
    ws.addEventListener('open', () => setWsConnected(true));
    ws.addEventListener('close', () => setWsConnected(false));

    return () => {
      ws.close();
    };
  }, [token]);

  // Refresh history helper
  const refreshHistory = async () => {
    if (!token) return;
    try {
      const h = await queryUserHistory(token, 10);
      if (h && h.records) {
        setRecentHistory(h.records);
      }
    } catch (error) {
      console.error("History refresh failed:", error);
    }
  };

  // Type guard pentru AnalysisResponse
  const isAnalysisResponse = (result: any): result is AnalysisResponse => {
    return result && typeof result === 'object' && 'verdict' in result;
  };

  // Initial Data Fetch
  useEffect(() => {
    if (!token) return;

    const fetchData = async () => {
      try {
        // 1. Dashboard Stats
        try {
          const s = await getDashboardStats(token);
          setStats(s);
        } catch (e) { 
          console.warn("Stats fetch failed (using defaults)", e);
          setStats({
            totalAnalyses: 0,
            todayAnalyses: 0,
            weeklyTrend: 0,
            deepfakesDetected: 0,
            authenticVideos: 0,
            inconclusiveResults: 0,
            averageProcessingTime: 0,
            weeklyData: []
          });
        }

        // 2. User History
        const h = await queryUserHistory(token, 10);
        if (h && h.records) {
          setRecentHistory(h.records);

          // 3. Extract Evidence from most recent completed job
          const lastCompleted = h.records.find(r => 
            r.verdict && ['FAKE', 'SUSPICIOUS', 'REAL'].includes(r.verdict)
          );
          
          if (lastCompleted?.job_id) {
            try {
              const jobDetails = await getJobStatus(lastCompleted.job_id, token);
              
              // Parse result if it's a string
              let result = jobDetails.result;
              if (typeof result === 'string') {
                try {
                  result = JSON.parse(result);
                } catch (e) {
                  console.error('Failed to parse result:', e);
                  result = undefined;
                }
              }

              // Type guard check
              if (result && isAnalysisResponse(result)) {
                // Extract evidence
                if (result.evidence) {
                  setArtifacts(result.evidence);
                }

                // Extract Gemini Vision annotated frames as evidence
                if (result.details?.gemini_vision?.annotated_frames && Array.isArray(result.details.gemini_vision.annotated_frames)) {
                  const frameEvidence: Evidence[] = result.details.gemini_vision.annotated_frames
                    .map((frame: any, idx: number) => {
                      const url = typeof frame === 'string' ? frame : (frame?.url || frame?.path || '');
                      if (!url) return null;
                      return {
                        type: "annotated_frame" as const,
                        title: `Frame ${idx + 1} Analysis`,
                        url,
                        description: result.details.gemini_vision.frame_analyses?.[idx]?.reasoning || "Gemini Vision frame analysis",
                        confidence: result.details.gemini_vision.frame_analyses?.[idx]?.confidence || 0
                      };
                    })
                    .filter(Boolean) as Evidence[];
                  
                  setArtifacts(prev => [...prev, ...frameEvidence]);
                }
              }

              // Calculate SynthID Score
              const totalRecent = h.records.length;
              const fakeRecent = h.records.filter(r => r.verdict === "FAKE").length;
              setSynthIDScore(totalRecent > 0 ? Math.round((fakeRecent / totalRecent) * 100) : 0);

            } catch (error) {
              console.error("Job details fetch failed:", error);
            }
          }
        }

      } catch (error) {
        console.error("Dashboard Load Error:", error);
      }
    };

    fetchData();
  }, [token]);

  // Calculate real-time stats
  const pendingCount = Array.from(activeJobs.values()).length;
  const avgProgress = pendingCount > 0 
    ? Math.round(Array.from(activeJobs.values()).reduce((a, b) => a + b, 0) / pendingCount)
    : 0;

  return (
    <MainLayout>
      <div className="flex flex-col h-full space-y-6">
        {/* HEADER */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
              Forensic Command Center
            </h1>
            <p className="text-muted-foreground mt-1 font-mono text-sm">
              OPERATOR: {user?.username?.toUpperCase()} // STATUS: ACTIVE
              {wsConnected && (
                <Badge variant="outline" className="ml-2 border-green-500 text-green-500">
                  🟢 LIVE
                </Badge>
              )}
            </p>
          </div>

          {/* Active Jobs Indicator */}
          {pendingCount > 0 && (
            <Card className="px-4 py-2 bg-blue-500/10 border-blue-500/20">
              <div className="flex items-center gap-3">
                <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
                <div>
                  <div className="text-xs text-muted-foreground">Active Analysis</div>
                  <div className="font-mono text-sm font-bold">{pendingCount} job{pendingCount > 1 ? 's' : ''}</div>
                </div>
                <Progress value={avgProgress} className="w-24" />
                <span className="text-xs font-mono">{avgProgress}%</span>
              </div>
            </Card>
          )}
        </div>

        {/* MAIN GRID */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">

          {/* 1. AGENT PULSE */}
          <div className="md:col-span-4">
            <AgentPulse />
          </div>

          {/* 2. STATS ROW */}
          <StatCard
            title="Threat Level"
            value={`${synthIDScore}%`}
            icon={Gauge}
            description="AI Content Detected (7d)"
            trend={synthIDScore > 50 ? "up" : synthIDScore > 0 ? "neutral" : "down"}
          />
          <StatCard
            title="Deepfakes Intercepted"
            value={stats?.deepfakesDetected.toString() || "0"}
            icon={Shield}
            description="Total Confirmed Fakes"
          />
          <StatCard
            title="Active Queue"
            value={pendingCount.toString()}
            icon={Activity}
            description="Jobs Processing"
            trend={pendingCount > 0 ? "up" : "neutral"}
          />
          <StatCard
            title="Weekly Volume"
            value={stats?.weeklyTrend.toString() || "0"}
            icon={FileVideo}
            description="Files Processed"
          />

          {/* 3. MIDDLE LAYER */}
          <div className="md:col-span-3 h-[400px]">
            <ForensicGallery artifacts={artifacts} />
          </div>

          <div className="md:col-span-1 h-[400px] flex flex-col gap-4">
            {/* SynthID Gauge */}
            <Card className="flex-1 bg-black/40 border-primary/10 flex items-center justify-center relative overflow-hidden">
              <div className="text-center z-10">
                <Gauge className={cn(
                  "h-16 w-16 mx-auto mb-2",
                  synthIDScore > 70 ? "text-red-500" : 
                  synthIDScore > 30 ? "text-yellow-500" : 
                  "text-green-500"
                )} />
                <div className="text-4xl font-bold">{synthIDScore}%</div>
                <div className="text-[10px] uppercase font-mono text-muted-foreground mt-1">
                  Deepfake Detection Rate
                </div>
              </div>
              <div className={cn(
                "absolute inset-0 opacity-20 blur-xl",
                synthIDScore > 70 ? "bg-red-500" : 
                synthIDScore > 30 ? "bg-yellow-500" : 
                "bg-green-500"
              )}></div>
            </Card>

            {/* System Status */}
            <Card className="flex-1 bg-primary/5 border-primary/20 p-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Celery Workers</span>
                  <Badge variant="outline" className="border-green-500 text-green-500">Online</Badge>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Gemini Vision</span>
                  <Badge variant="outline" className="border-green-500 text-green-500">Active</Badge>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Neo4j Graph</span>
                  <Badge variant="outline" className="border-green-500 text-green-500">Connected</Badge>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Redis Queue</span>
                  <Badge variant="outline" className="border-green-500 text-green-500">Ready</Badge>
                </div>
              </div>
            </Card>
          </div>

          {/* 4. BOTTOM LAYER */}
          <div className="md:col-span-4">
            <RecentAnalysisCard history={recentHistory} activeJobs={activeJobs} />
          </div>
        </div>
      </div>
    </MainLayout>
  );
}