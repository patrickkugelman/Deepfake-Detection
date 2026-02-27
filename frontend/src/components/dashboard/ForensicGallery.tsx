import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, ImageIcon, PlayCircle } from "lucide-react";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Evidence } from "@/api";
import { cn } from "@/lib/utils";

interface ForensicGalleryProps {
    artifacts: Evidence[];
}

export function ForensicGallery({ artifacts }: ForensicGalleryProps) {
    // Filter for visual artifacts
    const visualArtifacts = artifacts.filter(
        (a) => a.type === "frame" || a.type === "heatmap"
    );

    return (
        <Card className="col-span-1 md:col-span-3 border-primary/20 bg-muted/5 h-full">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-mono flex items-center gap-2">
                    <ImageIcon className="h-4 w-4 text-primary" />
                    FORENSIC_ARTIFACTS // EVIDENCE_POOL
                </CardTitle>
            </CardHeader>
            <CardContent>
                {visualArtifacts.length === 0 ? (
                    <div className="h-32 flex items-center justify-center text-muted-foreground text-sm border-2 border-dashed rounded-xl">
                        No visual artifacts detected in recent analysis.
                    </div>
                ) : (
                    <ScrollArea className="w-full whitespace-nowrap rounded-md border">
                        <div className="flex w-max space-x-4 p-4">
                            {visualArtifacts.map((artifact, i) => (
                                <div
                                    key={i}
                                    className="relative group w-[200px] aspect-video rounded-lg overflow-hidden border border-border bg-black"
                                >
                                    <img
                                        src={artifact.url}
                                        alt={artifact.title}
                                        className="object-cover w-full h-full transition-transform group-hover:scale-110 opacity-80 group-hover:opacity-100"
                                        onError={(e) => {
                                            (e.currentTarget as HTMLImageElement).style.display = 'none';
                                        }}
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent p-2 flex flex-col justify-end">
                                        <div className="flex justify-between items-center">
                                            <span className="text-[10px] font-bold text-white truncate max-w-[120px]">
                                                {artifact.title}
                                            </span>
                                            {artifact.confidence && (
                                                <span className={cn(
                                                    "text-[10px] px-1 rounded",
                                                    artifact.confidence > 80 ? "bg-red-500 text-white" : "bg-yellow-500 text-black"
                                                )}>
                                                    {(artifact.confidence).toFixed(0)}% FAKE
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    {/* Overlay Icon */}
                                    <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <AlertTriangle className="h-4 w-4 text-yellow-400" />
                                    </div>
                                </div>
                            ))}
                        </div>
                        <ScrollBar orientation="horizontal" />
                    </ScrollArea>
                )}
            </CardContent>
        </Card>
    );
}
