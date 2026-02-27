import { useState, useRef, useEffect, SyntheticEvent } from "react";
import { useLocation } from "react-router-dom";

// IMPORTURI UI - Asigură-te că aceste fișiere există în src/components/ui/
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import {
    MessageSquare, Minimize2, Bot, User,
    Loader2, FileImage, BarChart2, AlertTriangle,
    Paperclip, Send
} from "lucide-react";

// IMPORT UTILITAR - Asigură-te că fișierul este în src/lib/utils.ts
import { cn } from "@/lib/utils";

// IMPORT API & CONTEXT
import {
    chatWithBot,
    sendChatMessage, // Added import
    getJobStatus,
    getChatHistory,
    AnalysisResponse,
    Evidence,
    JobStatusResponse
} from "@/api";

import { useAuth } from "@/context/AuthContext";
import { toast } from "sonner";

interface Message {
    role: "user" | "assistant";
    content: string;
    evidence?: Evidence[];
    timestamp?: string;
}

export function ChatAssistant() {
    const { token } = useAuth();
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        { role: "assistant", content: "Salut! Sistemul Forensic Cortex este online. Pot explica de ce un fișier este considerat Fake sau Real." }
    ]);
    const [inputValue, setInputValue] = useState("");
    const [attachment, setAttachment] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);
    const location = useLocation();

    // Încărcare Istoric
    useEffect(() => {
        if (token && isOpen) {
            getChatHistory(token)
                .then((history: { messages: Message[] }) => {
                    if (history?.messages?.length > 0) {
                        setMessages(history.messages);
                    }
                })
                .catch((err: Error) => console.error("History fetch error:", err));
        }
    }, [token, isOpen]);

    // Scroll automant
    useEffect(() => {
        if (scrollRef.current) {
            const viewport = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
            if (viewport) (viewport as HTMLElement).scrollTop = viewport.scrollHeight;
        }
    }, [messages, isOpen]);

    const pollResponse = async (jobId: string) => {
        if (!token) return;

        const interval = setInterval(async () => {
            try {
                const stats: JobStatusResponse = await getJobStatus(jobId, token);

                if (stats.status === 'completed' && stats.result) {
                    clearInterval(interval);
                    setIsLoading(false);
                    const res = stats.result as AnalysisResponse;
                    setMessages(prev => [...prev, {
                        role: "assistant",
                        content: res.report || res.reasoning || "Analiză finalizată.",
                        evidence: res.evidence
                    }]);
                } else if (stats.status === 'failed') {
                    clearInterval(interval);
                    setIsLoading(false);
                    setMessages(prev => [...prev, { role: "assistant", content: "Eroare la procesare." }]);
                }
            } catch (err) {
                clearInterval(interval);
                setIsLoading(false);
            }
        }, 2000);
    };

    const handleSend = async () => {
        if ((!inputValue.trim() && !attachment) || isLoading) return;
        if (!token) return toast.error("Autentificare necesară");

        const userMsg = inputValue.trim();
        // Extract Context from Location (Results page)
        const currentFileHash = location.state?.analysis?.metadata?.file_hash;

        let currentJobId = location.state?.jobId;
        if (!currentJobId && location.pathname.includes("/results/")) {
            currentJobId = location.pathname.split("/results/")[1];
        }

        setMessages(prev => [...prev, { role: "user", content: userMsg || "Analiză fișier" }]);
        setInputValue("");
        setIsLoading(true);

        try {
            if (attachment) {
                // SCENARIO 1: Upload new file
                const jobRes = await chatWithBot(userMsg, token, currentFileHash, attachment);
                setAttachment(null);
                pollResponse(jobRes.job_id);
            } else if (currentJobId) {
                // SCENARIO 2: Chat about existing job (Q&A with visual evidence)
                const response = await sendChatMessage(currentJobId, userMsg, token);

                const evidenceList = (response.evidence || []).map((ev: any) => ({
                    type: ev.type || 'frame',
                    title: ev.title || 'Evidence',
                    url: ev.url,
                    description: ev.description || ''
                }));

                setMessages(prev => [...prev, {
                    role: "assistant",
                    content: response.content || "Nu am putut genera un răspuns.",
                    evidence: evidenceList
                }]);
                setIsLoading(false);
            } else {
                // SCENARIO 3: General Chat / New Job without file (fallback)
                const jobRes = await chatWithBot(userMsg, token);
                pollResponse(jobRes.job_id);
            }
        } catch (err) {
            console.error(err);
            toast.error("Cortex offline");
            setIsLoading(false);
        }
    };

    return (
        <>
            {!isOpen && (
                <Button
                    onClick={() => setIsOpen(true)}
                    className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-2xl z-50 bg-primary"
                >
                    <MessageSquare className="h-6 w-6 text-white" />
                </Button>
            )}

            {isOpen && (
                <div className="fixed bottom-4 right-4 w-[calc(100vw-2rem)] sm:w-[400px] h-[80vh] max-h-[600px] bg-card border border-border rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden animate-in slide-in-from-bottom-5">
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b bg-muted/20">
                        <div className="flex items-center gap-3">
                            <Bot className="h-5 w-5 text-primary" />
                            <span className="text-sm font-bold font-mono">CORTEX_LINK</span>
                        </div>
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setIsOpen(false)}>
                            <Minimize2 className="h-4 w-4" />
                        </Button>
                    </div>

                    <ScrollArea className="flex-1 p-4" ref={scrollRef}>
                        <div className="space-y-4">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={cn("flex gap-3", msg.role === "user" ? "flex-row-reverse" : "")}>
                                    <div className={cn("w-8 h-8 rounded-full flex items-center justify-center border",
                                        msg.role === "user" ? "bg-primary text-white" : "bg-muted text-primary")}>
                                        {msg.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                                    </div>
                                    <div className={cn("flex flex-col gap-2 max-w-[80%]", msg.role === "user" ? "items-end" : "")}>
                                        <div className={cn("p-3 rounded-2xl text-sm",
                                            msg.role === "user" ? "bg-primary text-white rounded-tr-none" : "bg-muted/50 border rounded-tl-none")}>
                                            {msg.content}
                                        </div>

                                        {msg.evidence?.map((ev, i) => (
                                            <Card key={i} className="bg-black/20 border-primary/10 overflow-hidden">
                                                <CardHeader className="p-2 pb-0">
                                                    <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-muted-foreground">
                                                        {ev.type === 'frame' ? <FileImage className="h-3 w-3" /> : <BarChart2 className="h-3 w-3" />}
                                                        {ev.title}
                                                    </div>
                                                </CardHeader>
                                                <CardContent className="p-2">
                                                    {ev.url && (
                                                        <img
                                                            src={ev.url}
                                                            alt="Evidence"
                                                            className="rounded mb-1"
                                                            // FIX: Error 7006 - Tipizarea parametrului 'e'
                                                            onError={(e: SyntheticEvent<HTMLImageElement>) => {
                                                                (e.currentTarget as HTMLImageElement).style.display = 'none';
                                                            }}
                                                        />
                                                    )}
                                                    <p className="text-[10px] text-slate-400 italic">{ev.description}</p>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>

                    {/* Input Area */}
                    <div className="p-4 border-t bg-background">
                        <form
                            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                            className="flex gap-2"
                        >
                            <input
                                type="file" id="chat-file" className="hidden"
                                onChange={(e) => setAttachment(e.target.files?.[0] || null)}
                            />
                            <Button
                                type="button" variant="ghost" size="icon"
                                className={cn(attachment ? "text-primary bg-primary/10" : "text-muted-foreground")}
                                onClick={() => document.getElementById('chat-file')?.click()}
                            >
                                <Paperclip className="h-4 w-4" />
                            </Button>
                            <Input
                                value={inputValue}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputValue(e.target.value)}
                                placeholder="Întreabă Cortex..."
                                className="bg-background"
                                disabled={isLoading}
                            />
                            <Button type="submit" size="icon" disabled={isLoading || (!inputValue.trim() && !attachment)}>
                                <Send className="h-4 w-4" />
                            </Button>
                        </form>
                    </div>
                </div>
            )}
        </>
    );
}