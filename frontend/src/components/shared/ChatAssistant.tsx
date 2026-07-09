import { useState, useRef, useEffect, useCallback } from "react";
import { useLocation, useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  MessageSquare, X, Send, Bot, User, Minimize2, Loader2,
  FileImage, BarChart2, AlertTriangle, Paperclip, Sparkles,
  ChevronDown, Shield, Zap, Eye, Maximize2, Copy, Check,
  Brain, Activity
} from "lucide-react";
import { cn } from "@/lib/utils";
import { chatWithBot, getJobStatus, sendChatMessage, getChatHistory, AnalysisResponse } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Evidence {
  type: "frame" | "heatmap" | "chart" | "alert" | "annotated_frame";
  title: string;
  url?: string;
  description: string;
  confidence?: number;
  verdict?: string;
  artifacts?: string[];
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  evidence?: Evidence[];
  timestamp?: Date;
  isTyping?: boolean;
  id: string;
}

// ─── Suggested Questions (context-aware) ────────────────────────────────────

const SUGGESTIONS_GENERAL = [
  "De ce a fost marcat ca FAKE?",
  "Ce artefacte au fost detectate?",
  "Cât de sigură este analiza?",
  "Explică-mi frame-ul cel mai suspect",
];

const SUGGESTIONS_FAKE = [
  "Unde exact sunt artefactele?",
  "Ce tip de deepfake este acesta?",
  "Arată-mi frame-urile FAKE",
  "Poate fi contestat verdictul?",
];

const SUGGESTIONS_REAL = [
  "De ce crezi că este autentic?",
  "Există vreun semn de editare?",
  "Ce indicatori confirmă autenticitatea?",
  "Analiza mai adânc imaginea",
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeId() {
  return Math.random().toString(36).slice(2);
}

function formatTime(d: Date) {
  return d.toLocaleTimeString('ro-RO', { hour: '2-digit', minute: '2-digit' });
}

function VerdictBadge({ verdict }: { verdict?: string }) {
  if (!verdict) return null;
  const v = verdict.toUpperCase();
  return (
    <span className={cn(
      "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold border",
      v === 'FAKE' ? 'bg-red-500/10 text-red-400 border-red-500/30' :
        v === 'REAL' ? 'bg-green-500/10 text-green-400 border-green-500/30' :
          'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
    )}>
      {v === 'FAKE' ? '⚠️' : v === 'REAL' ? '✅' : '❓'} {v}
    </span>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-primary/60 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.8s' }}
        />
      ))}
    </div>
  );
}

// Simple markdown-ish renderer (bold, lists, inline code)
function MessageContent({ content }: { content: string }) {
  const lines = content.split('\n');
  return (
    <div className="space-y-1 text-sm leading-relaxed">
      {lines.map((line, i) => {
        if (line.startsWith('### ')) return <p key={i} className="font-bold text-foreground text-xs uppercase tracking-wider mt-2">{line.slice(4)}</p>;
        if (line.startsWith('## ')) return <p key={i} className="font-bold text-foreground text-sm mt-2">{line.slice(3)}</p>;
        if (line.startsWith('**') && line.endsWith('**')) return <p key={i} className="font-semibold">{line.slice(2, -2)}</p>;
        if (line.startsWith('- ') || line.startsWith('• ')) {
          return <p key={i} className="flex gap-2"><span className="text-primary shrink-0 mt-0.5">▸</span><span dangerouslySetInnerHTML={{ __html: line.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} /></p>;
        }
        if (line.trim() === '') return <div key={i} className="h-1" />;
        return <p key={i} dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/`(.*?)`/g, '<code class="bg-muted px-1 rounded text-xs font-mono">$1</code>') }} />;
      })}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function ChatAssistant() {
  const { token } = useAuth();
  const location = useLocation();
  const { id: jobIdParam } = useParams();

  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: makeId(),
      role: "assistant",
      content: "Salut! Sunt **Cortex**, asistentul tău forensic AI.\n\nSunt specializat în analiza deepfake-urilor și pot explica în detaliu:\n- De ce un fișier este clasificat FAKE sau REAL\n- Ce artefacte specifice au fost detectate\n- Cum au lucrat agenții de analiză\n\nDacă ești pe o pagină de rezultate, ai context complet. Altfel, poți atașa un fișier pentru analiză directă.",
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [attachment, setAttachment] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Detect current job context
  const cleanJobId = jobIdParam?.replace("job-", "");
  const isOnResults = location.pathname.startsWith('/results') && !!cleanJobId;

  // Determine suggestions based on last message verdict
  const lastAssistantMsg = messages.filter(m => m.role === 'assistant').at(-1);
  const currentVerdict = lastAssistantMsg?.content.includes('FAKE') ? 'fake' :
    lastAssistantMsg?.content.includes('REAL') ? 'real' : 'general';
  const suggestions = currentVerdict === 'fake' ? SUGGESTIONS_FAKE :
    currentVerdict === 'real' ? SUGGESTIONS_REAL : SUGGESTIONS_GENERAL;

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isOpen]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Load persisted conversation for this job (survives page reloads) — without this,
  // every reload started a fresh chat with no memory of what was already discussed.
  useEffect(() => {
    if (!isOnResults || !cleanJobId || !token) return;
    let cancelled = false;
    (async () => {
      try {
        const { messages: stored } = await getChatHistory(cleanJobId, token);
        if (!cancelled && stored && stored.length > 0) {
          setMessages(stored.map((m) => ({
            id: makeId(),
            role: m.role as Message['role'],
            content: m.content,
            timestamp: m.ts ? new Date(m.ts) : new Date(),
          })));
          setShowSuggestions(false);
        }
      } catch {
        // best-effort — keep the default greeting if history can't be loaded
      }
    })();
    return () => { cancelled = true; };
  }, [isOnResults, cleanJobId, token]);

  const addTypingIndicator = () => {
    const id = makeId();
    setMessages(prev => [...prev, { id, role: 'assistant', content: '', isTyping: true, timestamp: new Date() }]);
    return id;
  };

  const removeTypingIndicator = (id: string) => {
    setMessages(prev => prev.filter(m => m.id !== id));
  };

  const addMessage = useCallback((msg: Omit<Message, 'id' | 'timestamp'>) => {
    setMessages(prev => [...prev, { ...msg, id: makeId(), timestamp: new Date() }]);
  }, []);

  const pollResponse = useCallback(async (jobId: string, typingId: string) => {
    if (!token) return;
    const interval = setInterval(async () => {
      try {
        const stats = await getJobStatus(jobId, token);
        if (stats.status === 'completed' && stats.result) {
          clearInterval(interval);
          removeTypingIndicator(typingId);
          setIsLoading(false);
          const response = stats.result as AnalysisResponse;
          addMessage({
            role: "assistant",
            content: response.report || response.reasoning || "Analiза completă. Nu am putut genera un răspuns detaliat.",
            evidence: response.evidence
          });
          setShowSuggestions(true);
        } else if (stats.status === 'failed') {
          clearInterval(interval);
          removeTypingIndicator(typingId);
          setIsLoading(false);
          addMessage({ role: "assistant", content: `❌ Eroare: ${stats.error || "Analiza a eșuat pe server."}` });
        }
      } catch {
        clearInterval(interval);
        removeTypingIndicator(typingId);
        setIsLoading(false);
        addMessage({ role: "assistant", content: "⚠️ Eroare de conexiune la server." });
      }
    }, 1500);
  }, [token, addMessage]);

  const handleSend = async (text?: string) => {
    const userMsg = (text || inputValue).trim();
    if ((!userMsg && !attachment) || isLoading || !token) return;

    setInputValue("");
    setShowSuggestions(false);
    addMessage({ role: "user", content: userMsg || `📎 ${attachment?.name}` });
    setIsLoading(true);

    const typingId = addTypingIndicator();

    try {
      if (isOnResults && cleanJobId && !attachment) {
        // Q&A about existing analysis — `messages` here is still the pre-send state
        // (setMessages above hasn't flushed yet), so this is exactly the prior turns.
        const historyForApi = messages
          .filter(m => !m.isTyping && (m.role === 'user' || m.role === 'assistant'))
          .map(m => ({ role: m.role, content: m.content }));
        const response = await sendChatMessage(cleanJobId, userMsg, token, historyForApi);
        removeTypingIndicator(typingId);
        setIsLoading(false);

        const evidenceList: Evidence[] = (response.evidence || []).map((ev: any) => ({
          type: ev.type || "frame",
          title: ev.title || "Evidence",
          url: ev.url,
          description: ev.description || "",
          confidence: ev.confidence,
          verdict: ev.verdict,
          artifacts: ev.artifacts
        }));

        addMessage({
          role: "assistant",
          content: response.content || "Nu am putut genera un răspuns.",
          evidence: evidenceList
        });
        setShowSuggestions(true);
      } else {
        // New file analysis
        const contextFileHash = location.state?.analysis?.metadata?.file_hash;
        const jobResponse = await chatWithBot(
          userMsg || "Analizează acest fișier",
          token,
          contextFileHash,
          attachment || undefined
        );
        setAttachment(null);
        pollResponse(jobResponse.job_id, typingId);
      }
    } catch (err: any) {
      removeTypingIndicator(typingId);
      setIsLoading(false);
      addMessage({ role: "assistant", content: `❌ Eroare: ${err.message || "Nu am putut contacta Cortex."}` });
      toast.error("Eroare la comunicarea cu Cortex.");
    }
  };

  const copyMessage = (content: string, id: string) => {
    navigator.clipboard.writeText(content);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const chatWidth = isExpanded ? "w-[700px]" : "w-[calc(100vw-2rem)] sm:w-[420px]";
  const chatHeight = isExpanded ? "h-[85vh]" : "h-[80vh] max-h-[660px]";

  return (
    <>
      {/* ── Floating Button ──────────────────────────────────────────────── */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-2xl z-50 animate-in fade-in zoom-in duration-300 bg-primary hover:bg-primary/90 transition-all hover:scale-110 flex items-center justify-center group"
        >
          <MessageSquare className="h-6 w-6 text-white" />
          {/* Pulse ring */}
          <span className="absolute inset-0 rounded-full bg-primary/30 animate-ping group-hover:hidden" />
          {/* Tooltip */}
          <span className="absolute right-16 bg-card border border-border text-foreground text-xs px-3 py-1.5 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
            Cortex AI — Asistent Forensic
          </span>
        </button>
      )}

      {/* ── Chat Window ──────────────────────────────────────────────────── */}
      {isOpen && (
        <div className={cn(
          "fixed bottom-4 right-4 bg-card border border-border rounded-2xl shadow-2xl z-50 flex flex-col animate-in slide-in-from-bottom-6 fade-in duration-300 transition-all",
          chatWidth, chatHeight
        )}>

          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-gradient-to-r from-primary/10 to-transparent rounded-t-2xl shrink-0">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="bg-primary/15 p-2 rounded-xl">
                  <Brain className="h-5 w-5 text-primary" />
                </div>
                <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-card animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-bold text-sm">Cortex AI</h3>
                  <Badge variant="outline" className="text-[9px] h-4 px-1.5 border-primary/30 text-primary">
                    <Sparkles className="h-2.5 w-2.5 mr-0.5" /> FORENSIC
                  </Badge>
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-green-500">
                  <Activity className="h-3 w-3" />
                  <span>Online • {isOnResults ? `Job: ${cleanJobId?.slice(0, 8)}...` : 'Mod General'}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground hidden sm:block"
                title={isExpanded ? "Micșorează" : "Mărește"}
              >
                <Maximize2 className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg hover:bg-muted transition-colors text-muted-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Context Banner (if on results page) */}
          {isOnResults && (
            <div className="px-4 py-2 bg-primary/5 border-b border-primary/10 shrink-0">
              <div className="flex items-center gap-2 text-xs text-primary">
                <Shield className="h-3.5 w-3.5 shrink-0" />
                <span>Ai context complet pentru această analiză — întreabă orice despre rezultate</span>
              </div>
            </div>
          )}

          {/* Messages */}
          <ScrollArea className="flex-1 px-4 py-3" ref={scrollRef}>
            <div className="space-y-4">
              {messages.map((msg) => (
                <div key={msg.id} className={cn("flex gap-2.5", msg.role === "user" ? "flex-row-reverse" : "")}>
                  {/* Avatar */}
                  <div className={cn(
                    "w-7 h-7 rounded-xl flex items-center justify-center shrink-0 mt-0.5",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-primary/10 text-primary border border-primary/20"
                  )}>
                    {msg.role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
                  </div>

                  {/* Content */}
                  <div className={cn("flex flex-col gap-1.5 max-w-[82%]", msg.role === "user" ? "items-end" : "items-start")}>
                    <div className={cn(
                      "px-3 py-2.5 rounded-2xl relative group",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-sm"
                        : "bg-muted/60 border border-border/60 rounded-tl-sm"
                    )}>
                      {msg.isTyping ? (
                        <TypingDots />
                      ) : (
                        <>
                          <MessageContent content={msg.content} />
                          {/* Copy button on hover */}
                          {msg.role === 'assistant' && msg.content && (
                            <button
                              onClick={() => copyMessage(msg.content, msg.id)}
                              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted"
                            >
                              {copied === msg.id
                                ? <Check className="h-3 w-3 text-green-400" />
                                : <Copy className="h-3 w-3 text-muted-foreground" />
                              }
                            </button>
                          )}
                        </>
                      )}
                    </div>

                    {/* Timestamp */}
                    {msg.timestamp && !msg.isTyping && (
                      <span className="text-[10px] text-muted-foreground px-1">
                        {formatTime(msg.timestamp)}
                      </span>
                    )}

                    {/* Evidence Cards */}
                    {msg.evidence && msg.evidence.length > 0 && (
                      <div className="w-full space-y-2 mt-1">
                        {msg.evidence.map((ev, i) => (
                          <div key={i} className="bg-background/80 border border-border/60 rounded-xl overflow-hidden">
                            <div className="flex items-center gap-2 px-3 py-2 bg-muted/30 border-b border-border/40">
                              {ev.type === 'frame' || ev.type === 'annotated_frame'
                                ? <FileImage className="h-3.5 w-3.5 text-blue-400 shrink-0" />
                                : ev.type === 'heatmap'
                                  ? <BarChart2 className="h-3.5 w-3.5 text-orange-400 shrink-0" />
                                  : <AlertTriangle className="h-3.5 w-3.5 text-red-400 shrink-0" />
                              }
                              <span className="text-xs font-semibold text-muted-foreground flex-1 truncate">{ev.title}</span>
                              <VerdictBadge verdict={ev.verdict} />
                            </div>

                            {ev.url && (
                              <div className="relative">
                                <img
                                  src={ev.url}
                                  alt={ev.title}
                                  className="w-full h-auto object-cover max-h-48"
                                  onError={e => { e.currentTarget.parentElement!.style.display = 'none'; }}
                                />
                              </div>
                            )}

                            <div className="px-3 py-2.5 space-y-1.5">
                              <p className="text-xs text-muted-foreground">{ev.description}</p>

                              {ev.confidence != null && (
                                <div className="flex items-center gap-2">
                                  <span className="text-[10px] text-muted-foreground">Confidence</span>
                                  <div className="flex-1 bg-muted rounded-full h-1">
                                    <div
                                      className={cn("h-1 rounded-full transition-all", ev.confidence > 70 ? "bg-red-400" : ev.confidence > 50 ? "bg-yellow-400" : "bg-green-400")}
                                      style={{ width: `${Math.min(100, ev.confidence)}%` }}
                                    />
                                  </div>
                                  <span className="text-[10px] font-mono font-bold">{ev.confidence.toFixed(0)}%</span>
                                </div>
                              )}

                              {ev.artifacts && ev.artifacts.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {ev.artifacts.slice(0, 3).map((a, j) => (
                                    <span key={j} className="text-[9px] bg-red-500/10 text-red-400 border border-red-500/20 px-1.5 py-0.5 rounded">
                                      {a}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          {/* Suggested Questions */}
          {showSuggestions && !isLoading && isOnResults && (
            <div className="px-4 py-2 border-t border-border/50 shrink-0">
              <p className="text-[10px] text-muted-foreground mb-1.5 flex items-center gap-1">
                <Zap className="h-3 w-3 text-primary" /> Întrebări sugerate
              </p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(s)}
                    className="text-[11px] px-2.5 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary hover:bg-primary/10 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <div className="px-4 py-3 border-t border-border bg-background/50 rounded-b-2xl shrink-0">
            {attachment && (
              <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-primary/10 rounded-lg border border-primary/20">
                <Paperclip className="h-3.5 w-3.5 text-primary shrink-0" />
                <span className="text-xs text-primary truncate flex-1">{attachment.name}</span>
                <button onClick={() => setAttachment(null)} className="text-muted-foreground hover:text-foreground">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            <form
              onSubmit={e => { e.preventDefault(); handleSend(); }}
              className="flex gap-2 items-center"
            >
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept="video/*,image/*,audio/*"
                onChange={e => setAttachment(e.target.files?.[0] || null)}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "p-2 rounded-xl transition-colors shrink-0",
                  attachment
                    ? "bg-primary/15 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
                title="Atașează fișier"
              >
                <Paperclip className="h-4 w-4" />
              </button>

              <Input
                ref={inputRef}
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                placeholder={
                  isOnResults
                    ? "Întreabă despre această analiză..."
                    : "Întreabă sau atașează un fișier..."
                }
                className="flex-1 rounded-xl border-border/60 bg-muted/30 text-sm"
                disabled={isLoading}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
                }}
              />

              <button
                type="submit"
                disabled={isLoading || (!inputValue.trim() && !attachment)}
                className={cn(
                  "p-2 rounded-xl transition-all shrink-0",
                  isLoading || (!inputValue.trim() && !attachment)
                    ? "bg-muted text-muted-foreground cursor-not-allowed"
                    : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm hover:shadow-md hover:scale-105"
                )}
              >
                {isLoading
                  ? <Loader2 className="h-4 w-4 animate-spin" />
                  : <Send className="h-4 w-4" />
                }
              </button>
            </form>

            <p className="text-[10px] text-muted-foreground text-center mt-2">
              Cortex • Powered by Gemini 2.0 Flash • DeepGuard Pro
            </p>
          </div>
        </div>
      )}
    </>
  );
}
