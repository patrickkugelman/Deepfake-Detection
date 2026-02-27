import { useEffect, useState, useMemo, useRef } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { MainLayout } from '@/layouts/MainLayout';
import { PageHeader } from '@/components/shared/PageHeader';
import { MarkdownRenderer } from '@/components/shared/MarkdownRenderer';

// UI & Icons
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import {
  Brain, Shield, AlertTriangle, RefreshCw,
  FileText, Activity, Loader2, CheckCircle2, Terminal, Fingerprint, Send, MessageCircle,
  Eye, Mic, Video, Type, Bot, Image
} from 'lucide-react';

// Utils & API
import { cn } from '@/lib/utils';
import { DEEPGUARD_API_URL } from '@/api';
import { useAuth } from '@/context/AuthContext';
import { useWebSocket } from '@/context/WebSocketContext';
import { JobDetails } from '@/types/analysis';

// Charts
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function Results() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { token } = useAuth();

  // Consumăm starea live din WebSocket
  const { messages, isConnected, analysisState } = useWebSocket();

  // State
  const [jobDetails, setJobDetails] = useState<JobDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Chatbot WebSocket
  const [chatWs, setChatWs] = useState<WebSocket | null>(null);
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  const sendChatMessage = (text: string) => {
    if (!text.trim() || !chatWs) return;
    chatWs.send(JSON.stringify({ message: text.trim() }));
    setChatMessages((prev) => [...prev, { role: 'user', content: text.trim() }]);
    setChatInput('');
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Poll for updates
  useEffect(() => {
    const fetchJobDetails = async () => {
      try {
        if (!token || !id) return;

        // Handle ID format (strip 'job-' if present)
        const cleanId = id.replace('job-', '');

        const response = await fetch(`${DEEPGUARD_API_URL}/jobs/${cleanId}`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) throw new Error('Failed to fetch job');

        const data = await response.json();
        setJobDetails(data);

        if (data.status === 'completed' || data.status === 'failed') {
          setLoading(false);
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchJobDetails();

    // Poll every 3s if not complete
    const interval = setInterval(() => {
      if (jobDetails?.status !== 'completed' && jobDetails?.status !== 'failed') {
        fetchJobDetails();
      }
    }, 3000);

    return () => clearInterval(interval);

  }, [id, token, jobDetails?.status]);

  // Chat WebSocket (only when analysis is complete)
  // Use a ref to track if we already connected — prevents re-render from killing the WS
  const chatWsConnected = useRef(false);
  useEffect(() => {
    const cleanId = id?.replace('job-', '') || '';
    if (!cleanId || jobDetails?.status !== 'completed') return;
    // Only connect ONCE
    if (chatWsConnected.current) return;
    chatWsConnected.current = true;

    const wsBase = (DEEPGUARD_API_URL || '').replace(/^http:\/\//, 'ws://').replace(/^https:\/\//, 'wss://');
    const wsUrl = `${wsBase}/ws/chat?job_id=${cleanId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setChatWs(ws);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'message') {
          setChatMessages((prev) => [...prev, { role: 'assistant', content: data.content }]);
        }
      } catch {
        // ignore
      }
    };

    ws.onerror = () => {
      chatWsConnected.current = false;
    };

    ws.onclose = () => {
      chatWsConnected.current = false;
      setChatWs(null);
    };

    return () => {
      ws.close();
      chatWsConnected.current = false;
      setChatWs(null);
    };
  }, [id, jobDetails?.status]);

  const formatFileSize = (bytes: number | null | undefined): string => {
    if (!bytes) return "N/A";
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(2)} ${units[unitIndex]}`;
  };

  const handleDownloadReport = async () => {
    if (!jobDetails) return;
    try {
      const response = await fetch(`${DEEPGUARD_API_URL}/generate_report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          filename: jobDetails.filename,
          verdict: jobDetails.verdict,
          confidence: (jobDetails.confidence || 0) * 100,
          report_text: jobDetails.reasoning,
          metadata: {
            file_size: jobDetails.file_size,
            resolution: jobDetails.resolution,
            mime_type: jobDetails.mime_type
          }
        })
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DeepGuard_Report_${id}.pdf`;
      a.click();
    } catch (error) {
      alert("Eroare la descărcarea raportului.");
    }
  };

  const isFake = jobDetails?.verdict?.toUpperCase().includes('FAKE');
  const isFinished = jobDetails?.status === 'completed';

  // --- LOADING STATE ---
  if (!isFinished && !jobDetails?.verdict) {
    return (
      <MainLayout>
        <div className="p-6 lg:p-8 max-w-4xl mx-auto space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h1 className="text-2xl font-bold flex items-center gap-3">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                Cortex analizează dovezile...
              </h1>
              <p className="text-sm text-muted-foreground italic">{analysisState.stage || jobDetails?.message || "Processing..."}</p>
            </div>
            <Badge variant={isConnected ? "secondary" : "destructive"} className="animate-pulse">
              {isConnected ? "CORTEX_LIVE_LINK" : "CONNECTING..."}
            </Badge>
          </div>

          <Progress value={jobDetails?.progress || analysisState.progress} className="h-2" />

          {/* Terminal View for Logs */}
          <Card className="bg-black border-slate-800 font-mono text-sm overflow-hidden shadow-2xl">
            <CardHeader className="border-b border-slate-800 bg-slate-900/50 py-3 px-4 flex flex-row items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/50" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
                <div className="w-3 h-3 rounded-full bg-green-500/50" />
              </div>
              <span className="ml-2 text-slate-500 text-xs">deepguard_cursor.log</span>
            </CardHeader>
            <CardContent className="p-6 h-[400px] overflow-y-auto space-y-2 text-slate-300 scrollbar-thin scrollbar-thumb-slate-800">
              {messages.map((m, i) => (
                <div key={i} className="flex gap-4 animate-in fade-in slide-in-from-left-2 duration-300">
                  <span className="text-slate-600 shrink-0">[{new Date(m.timestamp).toLocaleTimeString()}]</span>
                  <span className={cn(
                    m.status === 'thinking' ? 'text-blue-400 italic' :
                      m.status === 'success' ? 'text-green-400 font-bold' :
                        'text-slate-300'
                  )}>
                    {m.status === 'thinking' ? '🧠' : m.status === 'success' ? '✅' : 'ℹ️'} {m.message}
                  </span>
                </div>
              ))}
              <div className="flex gap-2 items-center text-primary animate-pulse italic pt-4">
                <Terminal className="h-4 w-4" />
                <span>Processing data stream...</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </MainLayout>
    );
  }

  // --- FINAL REPORT STATE ---
  return (
    <MainLayout>
      <div className="p-6 lg:p-8 max-w-6xl mx-auto space-y-8 animate-in fade-in duration-700">
        <PageHeader
          title="Expertiză Forensic Finalizată"
          description={`ID Referință: ${id}`}
          actions={
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => navigate('/upload')}>
                <RefreshCw className="h-4 w-4 mr-2" />Nouă Analiză
              </Button>
              <Button onClick={handleDownloadReport} className="bg-blue-600 hover:bg-blue-700">
                <FileText className="h-4 w-4 mr-2" />Raport PDF
              </Button>
            </div>
          }
        />

        {/* Verdict Hero Section */}
        <Card className={cn("border-2 shadow-2xl relative overflow-hidden transition-all duration-700",
          isFake
            ? 'border-red-500/40 bg-gradient-to-br from-red-950/30 via-red-900/10 to-background'
            : 'border-green-500/40 bg-gradient-to-br from-green-950/30 via-green-900/10 to-background'
        )} style={{
          boxShadow: isFake
            ? '0 0 60px rgba(239,68,68,0.15), 0 0 120px rgba(239,68,68,0.05)'
            : '0 0 60px rgba(34,197,94,0.15), 0 0 120px rgba(34,197,94,0.05)'
        }}>
          <div className="absolute top-0 right-0 p-4 opacity-[0.07]">
            <Shield className="h-40 w-40" />
          </div>
          {/* Animated pulse ring */}
          <div className={cn("absolute inset-0 rounded-xl animate-pulse opacity-[0.08]",
            isFake ? 'bg-red-500' : 'bg-green-500'
          )} style={{ animationDuration: '3s' }} />
          <CardContent className="p-10 text-center relative z-10">
            <div className={cn("inline-flex items-center justify-center p-5 rounded-full mb-6 border-2 shadow-lg",
              isFake
                ? 'bg-red-500/10 border-red-500/30 shadow-red-500/20'
                : 'bg-green-500/10 border-green-500/30 shadow-green-500/20'
            )}>
              {isFake
                ? <AlertTriangle className="h-12 w-12 text-red-500 drop-shadow-lg" />
                : <CheckCircle2 className="h-12 w-12 text-green-500 drop-shadow-lg" />}
            </div>
            <h2 className={cn("text-7xl font-black mb-2 tracking-tighter drop-shadow-lg",
              isFake ? 'text-red-500' : 'text-green-500')}>
              {jobDetails?.verdict?.toUpperCase() || "UNKNOWN"}
            </h2>

            <div className="flex flex-col items-center mt-4">
              <p className="text-muted-foreground uppercase tracking-[0.3em] text-xs font-bold">Nivel de Încredere Cortex</p>
              <div className={cn("text-5xl font-mono mt-2 font-bold",
                isFake ? 'text-red-400' : 'text-green-400')}>
                {(jobDetails?.confidence ? jobDetails.confidence * 100 : 0).toFixed(1)}%
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          <div className="xl:col-span-2 space-y-8">

            {/* AI Reasoning Report */}
            <Card className="bg-card/40 backdrop-blur-md border-border shadow-xl">
              <CardHeader className="border-b border-border">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Brain className="h-6 w-6 text-purple-500" /> Raționament Neuro-Simbolic
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-8 px-8 prose dark:prose-invert max-w-none prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-li:text-foreground prose-code:text-foreground">
                <MarkdownRenderer content={jobDetails?.reasoning || 'No reasoning available.'} />
              </CardContent>
            </Card>

            {/* Agent Breakdown */}
            {jobDetails?.result?.details && (
              <Card className="bg-card/40 backdrop-blur-md border-border shadow-xl">
                <CardHeader className="border-b border-border">
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Activity className="h-6 w-6 text-cyan-500" /> Rezultate per Agent
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-6 px-6 space-y-3">
                  {Object.entries(jobDetails.result.details).map(([agentName, agentData]: [string, any]) => {
                    if (!agentData || typeof agentData !== 'object') return null;
                    const verdict = String(agentData.verdict || 'N/A').toUpperCase();
                    if (verdict === 'N/A' || verdict === 'ERROR' || verdict === 'UNKNOWN') return null;
                    const fakeProbRaw = agentData.fake_probability ?? agentData.confidence ?? 0;
                    const fakeProb = typeof fakeProbRaw === 'number' ? (fakeProbRaw > 1 ? fakeProbRaw : fakeProbRaw * 100) : 0;
                    const agentIsFake = verdict === 'FAKE' || verdict === 'SUSPICIOUS';
                    const isSuspicious = verdict === 'SUSPICIOUS';

                    const iconMap: Record<string, any> = { vision: Eye, audio: Mic, video: Video, text: Type, gemini_vision: Bot, bio: Fingerprint };
                    const IconComp = iconMap[agentName] || Activity;

                    // Extract artifacts if available
                    const artifacts: string[] = agentData.artifacts_detected || agentData.anomalies || [];
                    const reason = agentData.reason || agentData.details || '';

                    return (
                      <div key={agentName} className={cn(
                        "rounded-xl border transition-all duration-300 overflow-hidden",
                        agentIsFake
                          ? 'border-red-500/20 bg-gradient-to-r from-red-950/20 to-transparent hover:border-red-500/40'
                          : isSuspicious
                            ? 'border-amber-500/20 bg-gradient-to-r from-amber-950/20 to-transparent hover:border-amber-500/40'
                            : 'border-green-500/15 bg-gradient-to-r from-green-950/10 to-transparent hover:border-green-500/30'
                      )}>
                        <div className="flex items-center gap-4 p-4">
                          <div className={cn("p-2.5 rounded-xl shadow-lg",
                            agentIsFake ? 'bg-red-500/15 shadow-red-500/10' : 'bg-green-500/15 shadow-green-500/10')}>
                            <IconComp className={cn("h-5 w-5", agentIsFake ? 'text-red-400' : 'text-green-400')} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-bold text-sm capitalize tracking-wide">{agentName.replace('_', ' ')}</span>
                              <Badge variant={agentIsFake ? 'destructive' : 'secondary'} className={cn("text-xs font-bold px-2.5 py-1",
                                isSuspicious && 'bg-amber-600 hover:bg-amber-700')}>
                                {verdict}
                              </Badge>
                            </div>
                            {/* Confidence bar */}
                            <div className="flex items-center gap-3">
                              <div className="flex-1 h-2.5 bg-muted rounded-full overflow-hidden relative">
                                {/* Threshold marker at 50% */}
                                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/20 z-10" />
                                <div
                                  className={cn("h-full rounded-full transition-all duration-700 ease-out",
                                    fakeProb > 70 ? 'bg-gradient-to-r from-red-600 to-red-400' :
                                    fakeProb > 40 ? 'bg-gradient-to-r from-amber-600 to-amber-400' :
                                    'bg-gradient-to-r from-green-600 to-green-400'
                                  )}
                                  style={{ width: `${Math.min(100, fakeProb)}%` }}
                                />
                              </div>
                              <span className={cn("text-sm font-mono font-bold w-16 text-right",
                                fakeProb > 70 ? 'text-red-400' : fakeProb > 40 ? 'text-amber-400' : 'text-green-400'
                              )}>{fakeProb.toFixed(1)}%</span>
                            </div>
                          </div>
                        </div>

                        {/* Expandable details area */}
                        {(reason || artifacts.length > 0) && (
                          <div className="px-4 pb-3 pt-0">
                            {reason && (
                              <p className="text-xs text-muted-foreground leading-relaxed mb-2 line-clamp-2" title={String(reason)}>
                                {String(reason)}
                              </p>
                            )}
                            {/* Artifact tags */}
                            {artifacts.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 mt-1">
                                {artifacts.slice(0, 5).map((art: string, i: number) => (
                                  <span key={i} className={cn(
                                    "inline-flex items-center text-[10px] font-medium px-2 py-0.5 rounded-full border",
                                    agentIsFake
                                      ? 'bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/20'
                                      : 'bg-muted text-foreground/70 border-border'
                                  )} title={art}>
                                    🔍 {art}
                                  </span>
                                ))}
                                {artifacts.length > 5 && (
                                  <span className="text-[10px] text-muted-foreground self-center">+{artifacts.length - 5} more</span>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </CardContent>
              </Card>
            )}

            {/* Evidence Gallery (Heatmaps & Annotated Frames) */}
            {(() => {
              const evidence: any[] = [];
              // Collect vision heatmaps (only those with a valid URL)
              const visionEvidence = jobDetails?.result?.details?.vision?.evidence;
              if (Array.isArray(visionEvidence)) {
                evidence.push(...visionEvidence.filter((e: any) => e.url && String(e.url).startsWith('/')));
              }
              // Collect Gemini annotated frames with per-frame analysis
              const geminiFrames = jobDetails?.result?.annotated_frames || jobDetails?.result?.details?.gemini_vision?.annotated_frames;
              const frameAnalyses = jobDetails?.result?.details?.gemini_vision?.frame_analyses || [];
              if (Array.isArray(geminiFrames)) {
                geminiFrames.forEach((frame: any, idx: number) => {
                  const url = typeof frame === 'string' ? frame : (frame.url || frame.path || '');
                  const analysis = frameAnalyses[idx] || {};
                  if (url) {
                    evidence.push({
                      type: 'frame',
                      title: `Frame ${idx + 1}`,
                      url,
                      description: analysis.reasoning || '',
                      verdict: analysis.verdict || '',
                      confidence: analysis.confidence || 0,
                      artifacts: analysis.artifacts_detected || [],
                    });
                  }
                });
              }

              if (evidence.length === 0) return null;

              return (
                <Card className="bg-card/40 backdrop-blur-md border-border shadow-xl">
                  <CardHeader className="border-b border-border">
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Image className="h-6 w-6 text-amber-500" /> Evidențe Vizuale
                      <span className="ml-auto text-xs font-normal text-muted-foreground">{evidence.length} {evidence.length === 1 ? 'evidență' : 'evidențe'}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6 px-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {evidence.map((item, idx) => {
                        const frameVerdict = String(item.verdict || '').toUpperCase();
                        const frameFake = frameVerdict === 'FAKE' || frameVerdict === 'SUSPICIOUS';
                        const frameReal = frameVerdict === 'REAL';
                        const frameInconclusive = frameVerdict === 'INCONCLUSIVE' || frameVerdict === '';
                        const frameConf = typeof item.confidence === 'number'
                          ? (item.confidence > 1 ? item.confidence : item.confidence * 100) : 0;

                        return (
                          <div key={idx} className={cn(
                            "group rounded-xl overflow-hidden border-2 transition-all duration-300 hover:shadow-xl relative",
                            frameFake
                              ? 'border-red-500/30 hover:border-red-500/60 shadow-red-500/5'
                              : frameReal
                                ? 'border-green-500/20 hover:border-green-500/40 shadow-green-500/5'
                                : 'border-yellow-500/20 hover:border-yellow-500/40'
                          )}>
                            {/* Image */}
                            <div className="relative bg-muted/50 aspect-video overflow-hidden">
                              <img
                                src={item.url?.startsWith('http') ? item.url : `${DEEPGUARD_API_URL}${item.url}`}
                                alt={item.title || `Evidence ${idx + 1}`}
                                className="w-full h-full object-contain opacity-95 group-hover:opacity-100 transition-opacity duration-300"
                                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                              />

                              {/* Verdict overlay badge (top-right) */}
                              {frameVerdict !== '' && (
                                <div className={cn(
                                  "absolute top-2 right-2 px-2.5 py-1 rounded-lg text-xs font-bold shadow-lg backdrop-blur-sm",
                                  frameFake
                                    ? 'bg-red-600/90 text-white border border-red-400/30'
                                    : frameReal
                                      ? 'bg-green-600/90 text-white border border-green-400/30'
                                      : 'bg-yellow-600/90 text-white border border-yellow-400/30'
                                )}>
                                  {frameFake ? '⚠️' : frameReal ? '✅' : '❓'} {frameVerdict || 'INCONCLUSIVE'}
                                </div>
                              )}

                              {/* Confidence badge (top-left) */}
                              {frameConf > 0 && (
                                <div className={cn(
                                  "absolute top-2 left-2 px-2 py-1 rounded-lg text-xs font-mono font-bold shadow-lg backdrop-blur-sm",
                                  frameConf > 70 ? 'bg-red-900/80 text-red-200 dark:bg-red-900/80 dark:text-red-200' :
                                  frameConf > 40 ? 'bg-amber-900/80 text-amber-200 dark:bg-amber-900/80 dark:text-amber-200' :
                                  'bg-black/70 text-white'
                                )}>
                                  {frameConf.toFixed(0)}%
                                </div>
                              )}

                              {/* Gradient overlay at bottom */}
                              <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/80 to-transparent" />
                            </div>

                            {/* Details below image */}
                            <div className="p-3 bg-muted/30">
                              <div className="flex items-center justify-between mb-1">
                                <p className="text-sm font-semibold text-foreground">{item.title || 'Evidence'}</p>
                              </div>
                              {item.description && (
                                <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">{item.description}</p>
                              )}
                              {/* Artifact tags for this frame */}
                              {item.artifacts && item.artifacts.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {item.artifacts.slice(0, 3).map((art: string, i: number) => (
                                    <span key={i} className={cn(
                                      "text-[10px] font-medium px-1.5 py-0.5 rounded-md border",
                                      frameFake
                                        ? 'bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/20'
                                        : 'bg-muted text-foreground/70 border-border'
                                    )}>
                                      {art}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              );
            })()}
          </div>

          {/* Sidebar Info */}
          <div className="space-y-8">
            {/* FILE DETAILS SECTION (NEW) */}
            <Card className="bg-card/40 border-border shadow-lg">
              <CardHeader><CardTitle className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Detalii Fișier</CardTitle></CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div className="flex flex-col gap-1 border-b border-border pb-2">
                  <span className="text-muted-foreground">Nume</span>
                  <span className="font-mono text-foreground text-xs break-all" title={jobDetails?.filename}>{jobDetails?.filename || "Unknown"}</span>
                </div>
                <div className="flex justify-between items-center border-b border-border pb-2">
                  <span className="text-muted-foreground">Format</span>
                  <span className="font-mono text-foreground">{jobDetails?.mime_type?.split('/')[1]?.split(';')[0]?.trim()?.toUpperCase() || "N/A"}</span>
                </div>
                <div className="flex justify-between items-center border-b border-border pb-2">
                  <span className="text-muted-foreground">Mărime</span>
                  <span className="font-mono text-foreground">{formatFileSize(jobDetails?.file_size)}</span>
                </div>
                <div className="flex justify-between items-center border-b border-border pb-2 last:border-0">
                  <span className="text-muted-foreground">Rezoluție</span>
                  <span className="font-mono text-foreground">{jobDetails?.resolution || "N/A"}</span>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-card/50 border-blue-500/20 shadow-lg">
              <CardHeader><CardTitle className="text-xs font-bold uppercase tracking-widest text-blue-500">Status Sistem</CardTitle></CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="text-[10px] uppercase font-bold text-muted-foreground mb-2 block">ID Job</label>
                  <code className="text-xs bg-muted p-2 rounded block break-all text-foreground">{jobDetails?.job_id}</code>
                </div>
                <div>
                  <label className="text-[10px] uppercase font-bold text-muted-foreground mb-2 block">Status</label>
                  <Badge variant={jobDetails?.status === 'completed' ? 'secondary' : 'outline'}>{jobDetails?.status?.toUpperCase()}</Badge>
                </div>
              </CardContent>
            </Card>

            {/* Chatbot Cortex */}
            <Card className="bg-card/50 border-purple-500/20 shadow-lg">
              <CardHeader>
                <CardTitle className="text-xs font-bold uppercase tracking-widest text-purple-500 flex items-center gap-2">
                  <MessageCircle className="h-4 w-4" /> Întreabă Cortex
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="h-48 overflow-y-auto space-y-2 text-sm border border-border rounded-lg p-3 bg-muted/30">
                  {chatMessages.length === 0 && (
                    <p className="text-muted-foreground text-xs italic">Pune întrebări despre rezumatul analizei.</p>
                  )}
                  {chatMessages.map((m, i) => (
                    <div key={i} className={cn("flex", m.role === 'user' ? 'justify-end' : 'justify-start')}>
                      <div className={cn(
                        "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                        m.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted text-foreground'
                      )}>
                        {m.content}
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
                <div className="flex gap-2">
                  <Input
                    placeholder="Ex: De ce e considerat fake?"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && sendChatMessage(chatInput)}
                  />
                  <Button size="icon" onClick={() => sendChatMessage(chatInput)} disabled={!chatWs}>
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}