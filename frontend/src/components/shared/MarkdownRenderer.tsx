import { cn } from "@/lib/utils";

interface MarkdownRendererProps {
    content: string;
    className?: string;
}

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
    if (!content) return null;

    const renderInline = (text: string, keyPrefix: string = '') => {
        // Parse inline: **bold**, ==highlight==, `code`, ~~strike~~
        const regex = /(\*\*[^*]+\*\*|==[^=]+=={1,2}|`[^`]+`|~~[^~]+~~)/g;
        const parts = text.split(regex);

        return parts.map((part, i) => {
            const key = `${keyPrefix}-${i}`;
            if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={key} className="text-foreground font-bold">{part.slice(2, -2)}</strong>;
            }
            if (part.startsWith('==') && part.endsWith('==')) {
                const inner = part.slice(2, -2);
                // Determine highlight color based on content
                const upperInner = inner.toUpperCase();
                if (upperInner.includes('FAKE') || upperInner.includes('DEEPFAKE') || upperInner.includes('MANIPUL')) {
                    return (
                        <mark key={key} className="bg-red-500/20 text-red-700 dark:text-red-300 px-1.5 py-0.5 rounded font-semibold border border-red-500/30" title="Detectat ca potențial fals">
                            {inner}
                        </mark>
                    );
                }
                if (upperInner.includes('REAL') || upperInner.includes('AUTENTIC') || upperInner.includes('AUTHENTIC')) {
                    return (
                        <mark key={key} className="bg-green-500/20 text-green-700 dark:text-green-300 px-1.5 py-0.5 rounded font-semibold border border-green-500/30" title="Confirmat autentic">
                            {inner}
                        </mark>
                    );
                }
                // Default yellow highlight
                return (
                    <mark key={key} className="bg-amber-500/20 text-amber-700 dark:text-amber-200 px-1.5 py-0.5 rounded font-medium border border-amber-500/30" title="Detaliu important">
                        {inner}
                    </mark>
                );
            }
            if (part.startsWith('`') && part.endsWith('`')) {
                return <code key={key} className="bg-muted text-cyan-700 dark:bg-slate-800 dark:text-cyan-300 px-1.5 py-0.5 rounded font-mono text-xs">{part.slice(1, -1)}</code>;
            }
            if (part.startsWith('~~') && part.endsWith('~~')) {
                return <del key={key} className="text-muted-foreground line-through">{part.slice(2, -2)}</del>;
            }
            return <span key={key}>{part}</span>;
        });
    };

    // Split into lines for block-level parsing
    const lines = content.split('\n');

    return (
        <div className={cn("prose dark:prose-invert max-w-none text-sm font-sans space-y-2", className)}>
            {lines.map((line, lineIdx) => {
                const trimmed = line.trim();

                // Headings
                if (trimmed.startsWith('### ')) {
                    return <h4 key={lineIdx} className="text-base font-bold text-foreground mt-4 mb-1">{renderInline(trimmed.slice(4), `h-${lineIdx}`)}</h4>;
                }
                if (trimmed.startsWith('## ')) {
                    return <h3 key={lineIdx} className="text-lg font-bold text-foreground mt-4 mb-1">{renderInline(trimmed.slice(3), `h-${lineIdx}`)}</h3>;
                }

                // Bullet points with custom styling
                if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || trimmed.startsWith('* ')) {
                    const bulletText = trimmed.slice(2);
                    const upperBullet = bulletText.toUpperCase();
                    let dotColor = 'bg-muted-foreground';
                    if (upperBullet.includes('FAKE') || upperBullet.includes('DEEPFAKE') || upperBullet.includes('ANOMAL') || upperBullet.includes('MANIPUL') || upperBullet.includes('ARTEFACT') || upperBullet.includes('ARTIFACT')) {
                        dotColor = 'bg-red-500';
                    } else if (upperBullet.includes('REAL') || upperBullet.includes('NATURAL') || upperBullet.includes('AUTHENT') || upperBullet.includes('NORMAL')) {
                        dotColor = 'bg-green-500';
                    } else if (upperBullet.includes('SUSPICI') || upperBullet.includes('WARNING') || upperBullet.includes('ATENȚI')) {
                        dotColor = 'bg-amber-500';
                    }
                    return (
                        <div key={lineIdx} className="flex items-start gap-2.5 pl-2">
                            <span className={cn("mt-2 w-2 h-2 rounded-full shrink-0", dotColor)} />
                            <span className="text-foreground/90 leading-relaxed">{renderInline(bulletText, `b-${lineIdx}`)}</span>
                        </div>
                    );
                }

                // Horizontal rule
                if (trimmed === '---' || trimmed === '***') {
                    return <hr key={lineIdx} className="border-border my-3" />;
                }

                // Empty lines → spacing
                if (trimmed === '') {
                    return <div key={lineIdx} className="h-1" />;
                }

                // Normal paragraph
                return <p key={lineIdx} className="text-foreground/90 leading-relaxed">{renderInline(trimmed, `p-${lineIdx}`)}</p>;
            })}
        </div>
    );
}
