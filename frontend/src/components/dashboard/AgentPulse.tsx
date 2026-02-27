import { Activity, Cpu, AlertCircle, CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

interface AgentStatus {
    name: string;
    status: "ONLINE" | "OFFLINE" | "BUSY" | "ERROR";
    latency: number;
}

const AGENTS = [
    { id: "text", label: "TEXT_CORE", icon: Cpu },
    { id: "audio", label: "AUDIO_SONAR", icon: Activity },
    { id: "video", label: "VIDEO_I3D", icon: Activity },
    { id: "bio", label: "BIO_SENSORS", icon: Activity },
    { id: "synthid", label: "SYNTHID_SENTINEL", icon: Cpu },
];

export function AgentPulse() {
    const [statuses, setStatuses] = useState<Record<string, AgentStatus>>({});

    // Mock polling for demonstration - in prod use WebSocket
    useEffect(() => {
        const interval = setInterval(() => {
            const newConfigs: Record<string, AgentStatus> = {};
            AGENTS.forEach((a) => {
                const isOnline = Math.random() > 0.1;
                newConfigs[a.id] = {
                    name: a.label,
                    status: isOnline ? "ONLINE" : "ERROR",
                    latency: Math.floor(Math.random() * 150) + 20,
                };
            });
            setStatuses(newConfigs);
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    return (
        <Card className="col-span-1 md:col-span-2 lg:col-span-3 border-primary/20 bg-muted/5">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-mono flex items-center gap-2">
                    <Activity className="h-4 w-4 text-primary animate-pulse" />
                    SYSTEM_STATUS // AGENT_PULSE
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {AGENTS.map((agent) => {
                        const status = statuses[agent.id]?.status || "ONLINE"; // Default to Online for demo
                        const latency = statuses[agent.id]?.latency || 45;
                        const isOnline = status === "ONLINE";

                        return (
                            <div
                                key={agent.id}
                                className={cn(
                                    "flex items-center justify-between p-3 rounded-xl border transition-all",
                                    isOnline
                                        ? "bg-black/20 border-border hover:border-primary/50"
                                        : "bg-destructive/10 border-destructive/50"
                                )}
                            >
                                <div className="flex items-center gap-3">
                                    <div className={cn("relative flex h-3 w-3")}>
                                        {isOnline && (
                                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                        )}
                                        <span className={cn(
                                            "relative inline-flex rounded-full h-3 w-3",
                                            isOnline ? "bg-emerald-500" : "bg-red-500"
                                        )}></span>
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
                                            {agent.label}
                                        </span>
                                        <span className="text-xs font-mono font-bold">
                                            {isOnline ? `${latency}ms` : "OFFLINE"}
                                        </span>
                                    </div>
                                </div>
                                {isOnline ? (
                                    <CheckCircle2 className="h-4 w-4 text-emerald-500 opacity-50" />
                                ) : (
                                    <AlertCircle className="h-4 w-4 text-destructive opacity-50" />
                                )}
                            </div>
                        );
                    })}
                </div>
            </CardContent>
        </Card>
    );
}
